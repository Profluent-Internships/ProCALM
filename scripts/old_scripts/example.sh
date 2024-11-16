CUDA_VISIBLE_DEVICES=1 python runner_old.py --model ec-onehot-swissprot-progen2large --checkpoint ba32964 --temp 0.3 --ec train+test --num_seqs 225

CUDA_VISIBLE_DEVICES=1 python runner_old.py --model ec-onehot-swissprot-progen2xlarge --checkpoint ba64974 --temp 0.3 --ec train+test --num_seqs 225 --batch_size 15