import argparse
import logging
import os

import torch
import torch.distributed as dist
from streaming.base.util import clean_stale_shared_memory
from torch.distributed.elastic.multiprocessing.errors import record
from progen_conditional.composer import get_trainer


def setup_dist():
    rank = int(os.environ.get("RANK", -1))
    if dist.is_available() and torch.cuda.is_available() and rank != -1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yml")
    parser.add_argument("--new-run", action="store_true", default=False)
    parser.add_argument("--disable-logging", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    return args

@record
def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__))) #change directory to the current directory

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)s:%(funcName)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    setup_dist()
    clean_stale_shared_memory()
    args = parse_args()

    trainer = get_trainer(
        args.config,
        force_new_run=args.new_run,
        disable_logging=args.disable_logging,
        debug=args.debug,
    )

    if trainer.state.max_duration >= trainer.state.timestamp.get(trainer.state.max_duration.unit):
        trainer.fit()
    
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
