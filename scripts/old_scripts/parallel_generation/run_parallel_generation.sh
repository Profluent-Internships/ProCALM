#!/bin/bash

MODEL=$1
CHECKPOINT=$2
TEMP=$3

CUDA_VISIBLE_DEVICES=0 python runner.py --model $MODEL --checkpoint $CHECKPOINT --temp $TEMP --ec batch1 --num_seqs 225 &
CUDA_VISIBLE_DEVICES=1 python runner.py --model $MODEL --checkpoint $CHECKPOINT --temp $TEMP --ec batch2 --num_seqs 225 &
CUDA_VISIBLE_DEVICES=2 python runner.py --model $MODEL --checkpoint $CHECKPOINT --temp $TEMP --ec batch3 --num_seqs 225 &
CUDA_VISIBLE_DEVICES=3 python runner.py --model $MODEL --checkpoint $CHECKPOINT --temp $TEMP --ec batch4 --num_seqs 225 &

wait