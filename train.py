import argparse
import os
import time
from transformers import T5Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import numpy as np

from utils.dataset import dataset
from utils.utils import logger, EarlyStopping, set_seed
from utils.gpt2 import GPT2Config, GPT2Model

def initialize(device,
            tokenizer: T5Tokenizer,
            ignore_index: int,
            len_train_iter: int,
            epochs: int = 20,
            lr: float = 1e-5,
            n_positions: int = 512,
            n_embed: int = 768,
            n_layer: int = 12,
            n_head: int = 8,
            n_inner: int = 3072,
            activation_func: str = 'gelu',
            resid_pdrop: float = 0.1,
            embed_pdrop: float = 0.1,
            attn_pdrop: float = 0.1,
            layer_norm_epsilon: float = 1e-5):

    config =  GPT2Config(vocab_size=tokenizer.vocab_size,
                        n_positions=n_positions,
                        n_embed=n_embed,
                        n_layer=n_layer,
                        n_head=n_head,
                        n_inner=n_inner,
                        activation_func=activation_func,
                        resid_pdrop=resid_pdrop,
                        embed_pdrop=embed_pdrop,
                        attn_pdrop=attn_pdrop,
                        layer_norm_epsilon=layer_norm_epsilon)

    model = GPT2Model(config=config).to(device)

    # freeze_layers = [
    #     f'transformer.h.{str(idx)}' for idx in range(num_freeze_layers)]
    # for name, param in model.named_parameters():
    #     if param.requires_grad and any(freeze_layer in name for freeze_layer in freeze_layers):
    #         param.requires_grad = False

    no_decay = ['bias', 'ln.weight', 'ln_1.weight', 'ln_2.weight']
    param_optimizer = [[name, param] for name,
                       param in model.named_parameters() if param.requires_grad]
    optimizer_grouped_parameters = [
        {'params': [param for name, param in param_optimizer if not any(nd in name for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [param for name, param in param_optimizer if any(nd in name for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    n_steps = len_train_iter * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_training_steps=n_steps, num_warmup_steps=100)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    return model, optimizer, scheduler, criterion, epochs


def step(model: GPT2Model,
        criterion: nn.CrossEntropyLoss,
        optimizer, scheduler,
        batch,
        device: torch.device,
        pad_idx: int):

    source_ids, target_ids, _ = tuple(t.to(device) for t in batch)
    # source_ids, target_ids shape of: [batch_size, seq_len]

    optimizer.zero_grad()

    # logits size of: [batch_size, seq_len, vocab_size]
    logits = model(input_ids=source_ids, pad_idx=pad_idx)[0]
    logits = logits[:, :-1].contiguous().view(-1, logits.size(-1))

    loss = criterion(logits, target_ids[:, 1:].contiguous().view(-1))

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

    optimizer.step()
    scheduler.step()

    return loss.item() 


def validate(model: GPT2Model,
            criterion: nn.CrossEntropyLoss,
            val_iter: torch.utils.data.DataLoader,
            device: torch.device,
            pad_idx: int):

    model.eval()

    running_loss = 0.0
    with torch.no_grad():
        for batch in val_iter:
            source_ids, target_ids, _ = tuple(t.to(device) for t in batch)

            logits = model(input_ids=source_ids, pad_idx=pad_idx)[0]
            logits = logits[:, :-1].contiguous().view(-1, logits.size(-1))

            loss = criterion(logits, target_ids[:, 1:].contiguous().view(-1))

            running_loss += loss.item()
    val_loss = running_loss/(len(val_iter))
    perplexity = torch.exp(torch.tensor(val_loss))

    return val_loss, perplexity


def train(model: GPT2Model,
        criterion: nn.CrossEntropyLoss,
        optimizer, scheduler,
        train_iter: torch.utils.data.DataLoader,
        val_iter: torch.utils.data.DataLoader,
        check_point: str,
        epochs: int, device: torch.device,
        pad_idx: int,
        patience: int = 5,
        delta: float = 1e-6):

    early_stopping = EarlyStopping(patience=patience, delta=delta)
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        start = time.time()
        model.train()
        running_loss = 0.0

        for idx, batch in enumerate(train_iter):

            loss = step(model, criterion, optimizer, scheduler, batch, device, pad_idx)
            running_loss += loss

            if (idx+1) % 100 == 0 or idx == 0:
                print("Epoch: {}/{} - iter: {}/{} - train_loss: {}".format(epoch + 1, epochs, idx+1, len(train_iter), running_loss/(idx+1)))
        else:
            train_loss = running_loss/(len(train_iter))
            print("Epoch: {}/{} - iter: {}/{} - train_loss: {}\n".format(epoch +
                  1, epochs, idx + 1, len(train_iter), train_loss))

            print('Evaluating...')
            val_loss, perplexity = validate(model, criterion, val_iter, device, pad_idx)
            print("    Val loss: {} - perplexity: {}\n".format(val_loss, perplexity))

            logger.info(f"Saving model to {os.path.join(check_point, 'cp'+str(epoch+1)+'.pt')}")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss
            }, os.path.join(check_point, 'cp'+str(epoch+1)+'.pt'))

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            early_stopping(val_loss)
            if early_stopping.early_stop:
                logger.info(
                    f"Early stopping. Saving log loss in {os.path.join(check_point, 'log_loss.txt')}")
                break
            logger.info(f"Total time per epoch: {time.time()-start} seconds")
    train_losses, val_losses = np.array(train_losses).reshape(-1, 1), np.array(val_losses).reshape(-1, 1)
    np.savetxt(os.path.join(check_point, 'log_loss.txt'), np.hstack((train_losses, val_losses)), delimiter='#')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='./data', help='data root dir')
    parser.add_argument('--file_name', type=str, default='jp_text_sum.csv', help='data file name')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--max_seq_len', type=int, default=512, help='max sequence length')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint', help='dir to save checkpoints')
    parser.add_argument('--patience', type=int, default=5, help='Patience early stopping')
    parser.add_argument('--delta', type=float, default=1e-6, help='Delta early stopping')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path to resume')
    args = parser.parse_args()

    ROOT_DIR = args.root_dir
    FILE_NAME = args.file_name
    BATCH_SIZE = args.batch_size
    MAX_LEN = args.max_seq_len
    EPOCHS = args.epochs
    LR = args.lr
    CHECKPOINT = args.checkpoint
    PATIENCE = args.patience
    DELTA = args.delta

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    set_seed()

    logger.info(f"Loading T5Tokenizer for 'rinna/japanese-gpt2-medium' model...")
    tokenizer = T5Tokenizer.from_pretrained('rinna/japanese-gpt2-medium')
    tokenizer.do_lower_case = True
    ignore_index = tokenizer.pad_token_id

    train_iter = dataset(tokenizer=tokenizer,
                         mode='train',
                         root=ROOT_DIR,
                         file_name=FILE_NAME,
                         max_len=MAX_LEN,
                         batch_size=BATCH_SIZE)

    val_iter = dataset(tokenizer=tokenizer,
                       mode='valid',
                       root=ROOT_DIR,
                       file_name=FILE_NAME,
                       max_len=MAX_LEN,
                       batch_size=BATCH_SIZE)

    logger.info('Initializing model...\n')
    model, optimizer, scheduler, criterion, epochs = initialize(device=DEVICE,
                                                                tokenizer=tokenizer,
                                                                ignore_index=ignore_index,
                                                                len_train_iter=len(train_iter),
                                                                epochs=EPOCHS,
                                                                lr=LR)
    if args.resume:
        ckp = torch.load(args.resume)
        model.load_state_dict(ckp['model_state_dict'])
        optimizer.load_state_dict(ckp['optimizer_state_dict'])
        scheduler.load_state_dict(ckp['scheduler_state_dict'])

    logger.info(f"Saving config to: {os.path.join(CHECKPOINT, 'config.pt')}")
    torch.save({
        'max_len': MAX_LEN,
        'batch_size': BATCH_SIZE,
        'ignore_index': ignore_index,
        'len_train_iter': len(train_iter),
        'epochs': epochs,
        'lr': LR
    }, os.path.join(CHECKPOINT, 'config.pt'))

    logger.info('Training model...\n')
    logger.info(f'Using lr = {LR}')

    start = time.time()
    train(model=model,
          criterion=criterion,
          optimizer=optimizer,
          scheduler=scheduler,
          train_iter=train_iter,
          val_iter=val_iter,
          check_point=CHECKPOINT,
          epochs=EPOCHS,
          device=DEVICE,
          pad_idx=ignore_index,
          patience=PATIENCE,
          delta=DELTA)
    logger.info(f'Total time: {(time.time()-start)} seconds')


if __name__ == '__main__':
    main()
