#!/bin/bash

MODEL=$1
CHECKPOINT=$2
TEMP=$3

CUDA_VISIBLE_DEVICES=0 python runner.py --model $MODEL --checkpoint $CHECKPOINT --temp $TEMP --tax 2.1224.1236.2887326.468.469.470 --num_seqs 900 &
CUDA_VISIBLE_DEVICES=1 python runner.py --model $MODEL --checkpoint $CHECKPOINT --temp $TEMP --tax 2157.2283796.183967.2301.46630.46631.46632 --num_seqs 900 &
CUDA_VISIBLE_DEVICES=2 python runner.py --model $MODEL --checkpoint $CHECKPOINT --temp $TEMP --tax 2759.4890.147550.5125.5129.5543.51453 --num_seqs 900 &
CUDA_VISIBLE_DEVICES=3 python runner.py --model $MODEL --checkpoint $CHECKPOINT --temp $TEMP --tax 10239.2731618.2731619.-1.2946170.10663.-1 --num_seqs 900 &

wait