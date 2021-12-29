import random
import numpy as np
import torch
import os
import argparse
from transformers import T5Tokenizer, top_k_top_p_filtering
import torch
import torch.nn as nn
from train import initialize

from utils.utils import create_logger

logger = create_logger()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    logger.info(f"using seed: {seed}")


def summary(text: str,
            device: torch.device,
            model,
            pad_idx: int,
            tokenizer: T5Tokenizer,
            max_seq_len: int = 512,
            summary_max_len: int = 96) -> str:

    tokens = tokenizer.encode(text=text,
                              max_length=max_seq_len,
                              truncation=True)[:-1] + [tokenizer.sep_token_id]
    
    tokens = torch.tensor(tokens).to(device).unsqueeze(0)

    sep_idx = len(tokens)-1
    with torch.no_grad():
        for _ in range(summary_max_len):
            last_logit = model(input_ids=tokens, pad_idx=pad_idx, device=device).logits[:, -1]

            filter = top_k_top_p_filtering(last_logit, top_k=50, top_p=1.0)
            props = nn.functional.softmax(filter, dim=-1)
            final_token = torch.multinomial(props, num_samples=1)
            if final_token[0, 0].cpu().numpy() == tokenizer.eos_token_id:
                return tokenizer.decode(tokens.tolist()[0][sep_idx:])

            tokens = torch.cat([tokens, final_token], dim=-1)
        return tokenizer.decode(tokens.tolist()[0][sep_idx:])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--summary_max_len', type=int, default=64, help='number of summary tokens will be generated')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint')
    parser.add_argument('--max_seq_len', type=int, default=512)
    args = parser.parse_args()

    SUMMARY_MAX_LEN = args.summary_max_len
    CHECK_POINT = args.checkpoint
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    set_seed()
    config_file = os.path.join('/'.join(CHECK_POINT.split('/')[:-1]), 'config.pt')
    logger.info(f"Loading config file from: {config_file}...")
    config = torch.load(config_file)

    if args.max_seq_len is not None:
        max_len = args.max_seq_len
    else:
        max_len = config['max_len']
    ignore_index = config['ignore_index']
    len_train_iter = config['len_train_iter']
    epochs = config['epochs']
    lr = config['lr']

    logger.info(f"Loading T5Tokenizer for 'rinna/japanese-gpt2-medium' model...")
    tokenizer = T5Tokenizer.from_pretrained('rinna/japanese-gpt2-medium')
    tokenizer.do_lower_case = True

    logger.info("Initializing model...\n")
    model, _, _, _, _ = initialize(device=DEVICE, tokenizer=tokenizer, ignore_index=ignore_index,
                                len_train_iter=len_train_iter, epochs=epochs, lr=lr)
    
    
    while True:
        text = input("Enter text here: ")
        s = summary(text=text,
                    device = DEVICE,
                    model=model,
                    pad_idx=ignore_index,
                    tokenizer=tokenizer,
                    max_seq_len=max_len,
                    summary_max_len=SUMMARY_MAX_LEN)
        print(s)


if __name__ == '__main__':
    main()
