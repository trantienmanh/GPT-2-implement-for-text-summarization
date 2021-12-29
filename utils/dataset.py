import os
import pandas as pd
from torch.utils.data import DataLoader, IterableDataset
from utils.utils import create_logger

logger = create_logger()


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
            tokens = self.tokenizer.encode(text=source + self.tokenizer.sep_token + target,
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.max_len,
                                    return_tensors='pt')
            yield tokens[0], tokens[0]


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