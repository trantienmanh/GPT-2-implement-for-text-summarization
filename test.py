import random
import os
import numpy as np
import torch
import argparse
from transformers import T5Tokenizer
import torch
from utils.dataset import dataset

from utils.utils import create_logger
from train import initialize, validate

logger = create_logger()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    logger.info(f"using seed: {seed}")

# def validate(model, criterion, val_iter, device):
#     model.eval()

#     running_loss = 0.0
#     preds, targets = [], []
#     with torch.no_grad():
#         for batch in val_iter:
#             input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)

#             logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
#             logits = logits[:, :-1]
#             preds.extend(torch.argmax(logits, dim=-1).detach().cpu().numpy())
#             labels = labels[:, 1:]
#             targets.extend(labels)

#             loss = criterion(logits.contiguous().view(-1, logits.size(-1)), labels.contiguous().view(-1))

#             running_loss += loss.item()
#     val_loss = running_loss/(len(val_iter))
#     perplexity = torch.exp(torch.tensor(val_loss))
#     # np.savetxt('./data/preds.txt', preds.astype(str))
#     # np.savetxt('./data/targets.txt', targets.astype(str))

#     return val_loss, perplexity

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, default='./data', help='data root dir')
    parser.add_argument('--file_name', type=str, default='jp_text_sum.csv', help='data file name')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint/', help='path to the best checkpoint')

    args = parser.parse_args()

    ROOT_DIR = args.root_dir
    FILE_NAME = args.file_name
    CHECK_POINT = args.checkpoint
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    set_seed()
    config_file = os.path.join('/'.join(CHECK_POINT.split('/')[:-1]), 'config.pt')
    logger.info(f"Loading config file from: {config_file}...")
    config = torch.load(config_file)

    batch_size = config['batch_size']
    max_len = config['max_len']
    ignore_index = config['ignore_index']
    len_train_iter = config['len_train_iter']
    epochs = config['epochs']
    lr = config['lr']

    logger.info(f"Loading T5Tokenizer for 'rinna/japanese-gpt2-medium' model...")
    tokenizer = T5Tokenizer.from_pretrained('rinna/japanese-gpt2-medium')
    tokenizer.do_lower_case = True

    test_iter = dataset(tokenizer=tokenizer,
                        mode='test',
                        root=ROOT_DIR,
                        file_name=FILE_NAME,
                        max_len=max_len,
                        batch_size=batch_size)

    logger.info("Initializing model...\n")
    model, _, _, criterion, _ = initialize(device=DEVICE,
                                        tokenizer=tokenizer,
                                        ignore_index=ignore_index,
                                        len_train_iter=len_train_iter,
                                        epochs=epochs,
                                        lr=lr)
    logger.info(f'Loading checkpoint from: {CHECK_POINT}')
    ckp = torch.load(CHECK_POINT)
    model.load_state_dict(ckp['model_state_dict'])

    logger.info('Testing model...\n')
    test_loss, perplexity = validate(model, criterion, test_iter,
                                    device=DEVICE,
                                    pad_idx=ignore_index)

    logger.info(f"Test loss: {test_loss} - perplexity: {perplexity}")

if __name__ == '__main__':
    main()