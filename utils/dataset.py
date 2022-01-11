import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset
from utils.utils import logger


class JPIterDataset(IterableDataset):
    def __init__(self,
                 tokenizer,
                 max_len: int=512,
                 random_state: int = 42,
                 mode='train',
                 root: str = './',
                 file_name: str = 'japanese_text_sum.csv') -> None:

        super(JPIterDataset, self).__init__()

        self.mode = mode
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = pd.read_csv(os.path.join(root, file_name))
        if mode == 'train':
            self.data = self.data[self.data.is_train]
            self.data = self.data.sample(frac=1, random_state=random_state)
        elif mode == 'valid':
            self.data = self.data[self.data.is_val]
        else:
            self.data = self.data[self.data.is_test]

    def __len__(self) -> int:
        return len(self.data)
    
    def __iter__(self):
        for source, target in zip(self.data.source, self.data.target):
            tmp = self.tokenizer.encode(self.additional_text)[:-1]
            src_tokens = self.tokenizer.encode(source)[:-1]
            trg_tokens = self.tokenizer.encode(target)

            input_ids = src_tokens + [self.tokenizer.sep_token_id] + tmp + trg_tokens

            if len(input_ids) > self.max_len:
                input_ids = input_ids[-512:]
                sep_idx = input_ids.index(self.tokenizer.sep_token_id)

                yield torch.tensor(input_ids), torch.tensor(input_ids), torch.tensor(sep_idx)
            else:
                pad_len = self.max_len - len(input_ids)
                input_ids = input_ids + [self.tokenizer.pad_token_id]* pad_len
                sep_idx = input_ids.index(self.tokenizer.sep_token_id)

                yield torch.tensor(input_ids), torch.tensor(input_ids), torch.tensor(sep_idx)


def dataset(tokenizer,
            mode='train',
            root: str = './data',
            file_name: str = 'jp_text_sum_extend.csv',
            max_len: int=512,
            batch_size: int = 4) -> DataLoader:

    if mode not in ['train', 'valid', 'test']:
        raise ValueError(
            "`mode` must be in the values: 'train', 'valid', or 'test'")

    logger.info(f"Creating {mode} iter dataset...")
    tensors = JPIterDataset(tokenizer=tokenizer,
                            max_len=max_len,
                            mode=mode,
                            root=root,
                            file_name=file_name)
    logger.info(f'Creating {mode} loader...')
    iterator = DataLoader(tensors, batch_size=batch_size)
    logger.info("Done!")
    return iterator