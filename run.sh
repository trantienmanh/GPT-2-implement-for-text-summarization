#!/bin/sh

TYPE_TRAIN=$1
RESUME_CKP=$2

if [-d checkpoint]; then mkdir checkpoint; fi

if [-d log]; then mkdir log; fi

if  [-d data]; then mkdir data; fi

if [$TYPE_TRAIN=="scratch"]
then
    python3 train.py --root_dir ./data/ --file_name jp_text_sum_extend.csv --batch_size 2 --max_seq_len 512 --epochs 20 --lr 3e-4 --checkpoint ./checkpoint/ --patience 5 --delta 1e-6
elif [$TYPE_TRAIN=="resume"]
then
    python3 train.py --root_dir ./data/ --file_name jp_text_sum_extend.csv --batch_size 2 --max_seq_len 512 --epochs 20 --lr 3e-4 --checkpoint ./checkpoint/ --patience 5 --delta 1e-6 --resume $RESUME_CKP
else
    echo "$TYPE_TRAIN not in [scratch, resume]"