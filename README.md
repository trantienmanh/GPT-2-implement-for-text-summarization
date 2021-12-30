# Implement GPT-2 from scratch for text summarization

## Directory
    ├── checkpoint/
    ├── log/
    ├── data/
    │   ├── jp_text_sum_extend.csv
    ├── utils/
    │   ├── __init__.py
    │   ├── dataset.py
    │   ├── gpt2.py
    │   ├── utils.py
    ├── train.py
    ├── test.py
    └── inference.py

## Install dependencies and make necessary folders:
```shell
cd GPT-2-implement-for-text-summarization
mkdir checkpoint log data
```
First, you must install dependencies, run the following command:
```shell
pip install -r requirements.txt
```

## Data
You can download processed data from [here](https://drive.google.com/file/d/1WKmIu7WIXGcroURhKYUukbO_5D_zGiOZ/view?usp=sharing), or raw text from [here](https://drive.google.com/file/d/1ZaKB5q6UN_3XGCj-jo-9Q-j-koUqaDol/view?usp=sharing), then put them to <b>./data</b> folder
## Training

For training from scratch, you can run command like this:

```shell
python3 train.py --root_dir ./data/ --file_name jp_text_sum_extend.csv --batch_size 2 --max_seq_len 512 --epochs 20 --lr 3e-4 --checkpoint ./checkpoint/ --patience 5 --delta 1e-6
```

For resume with the checkpoint, code may be:
```shell
python3 train.py --root_dir ./data/ --file_name jp_text_sum_extend.csv --batch_size 2 --max_seq_len 512 --epochs 20 --lr 3e-4 --checkpoint ./checkpoint/ --patience 5 --delta 1e-6 --resume path-to-the-checkpoint-is-resumed
```

## Testing

For evaluation, the command may like this:

```shell
python3 test.py --root_dir ./data/ --file_name jp_text_sum_extend.csv --checkpoint path-to-the-best-checkpoint
```

## Inference
Generate text with your model:
```shell
python3 inference.py --summary_max_len 64 --checkpoint path-to-the-best-checkpoint --max_seq_len 512
```