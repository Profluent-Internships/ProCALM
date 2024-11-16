import os
import random
from typing import List, Union
from ruamel.yaml import YAML
import copy
import argparse
import json
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from transformers import GenerationConfig
from progen_conditional.model import ProgenConditional #ProGenForCausalLM,
from progen_conditional.data import get_tokenizer, PAD_TOKEN_ID

CKPT_DIR = "results/"

class Runner():
    """
    Class for running generation on trained checkpoints with conditional adapters.
    """

    def __init__(self, model_name, checkpoint_name="ba11000", device="cuda") -> None:
        #load the training config to determine how to load the trained model
        self.model_name = model_name
        self.checkpoint_name = checkpoint_name
        self.device = device
        
        if model_name == 'progen2-base':
            #load a base progen model for generation
            self.model = ProGenForCausalLM.from_pretrained("data/pretrained_models/progen2-base")
            self.model.to(device)
            self.model.eval()
            self.tokenizer = get_tokenizer()
            self.pad_token_id = PAD_TOKEN_ID
            self.ec2encoding = None
            self.model_dir = os.path.join(CKPT_DIR, model_name)
            os.makedirs(self.model_dir, exist_ok=True)
        else:
            #load a model with conditional adapters
            self.model_dir = os.path.join(CKPT_DIR, model_name)
            ckpt_file = os.path.join(self.model_dir, 'huggingface', checkpoint_name)

            #load the huggingface model if it exists
            self.generation_config = GenerationConfig.from_pretrained(ckpt_file)
            self.model = ProgenConditional.from_pretrained(ckpt_file)
            self.progenconditional_config = self.model.config
            self.model.to(device)
            self.model.eval()

            self.tokenizer = get_tokenizer()
            self.pad_token_id = PAD_TOKEN_ID

            #load the dictionary mapping EC to encoding
            self.encoding_dicts = {}
            for key, encoding_file in self.progenconditional_config.encoding_files.items():
                self.encoding_dicts[key] = torch.load(encoding_file)
        
        np.random.seed(42)
        random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
    
    def sample(
        self,
        name: str, #name of the prompt
        conditions: Dict[str, str] = None, #dictionary mapping the type of condition (ec, tax) to the string specifying the condition
        context= '1', #start token
        num_return_sequences=45, #effectively the batch size
        temperature=0.3, #0.5 and 1 are a bit worse in performance
        top_p=0.95, #0.9 or 0.95
        max_length=1024,
        use_level1=False #whether to use the first level of the hierarchy
    ):
        """
        Runs one batch of generation with the specified conditions.
        """
        self.temp = 'temp' + str(temperature)
        self.name = name

        #check if the conditions are in the encoding dicts used to train the model. If not, do unconditional generation for that condition.
        
        print(f"Generating sequences for prompt ")

        condition_encodings = {}
        for key, encoding_dict in self.encoding_dicts.items():
            condition = conditions.get(key, None)

            if condition is not None:
                condition_encodings[key] = encoding_dict[condition].to(self.device)
            else:
                condition_encodings[key] = torch.zeros(1, self.progenconditional_config.encoding_dimensions[key]).to(self.device)
                
        #running things packaged into the huggingface class (alternatively could use beam search instead of probabilistic decoding)
        with torch.no_grad():
            input_ids = torch.tensor(self.tokenizer.encode(context).ids).view([1, -1]).to(self.device)

            if self.model_name == 'progen2-base':
                tokens_batch = self.model.generate(input_ids=input_ids, do_sample=True, temperature=temperature, max_length=max_length, top_p=top_p, num_return_sequences=num_return_sequences, pad_token_id=self.pad_token_id, eos_token_id=4)
            else:
                tokens_batch = self.model.generate(input_ids=input_ids, condition_encodings=condition_encodings, do_sample=True, temperature=temperature, max_length=max_length, top_p=top_p, num_return_sequences=num_return_sequences, pad_token_id=self.pad_token_id, eos_token_id=4)

            as_lists = lambda batch: [batch[i, ...].detach().cpu().numpy().tolist() for i in range(batch.shape[0])]
            self.sequences = self.tokenizer.decode_batch(as_lists(tokens_batch))

            return self.sequences
    
    def save_seqs(self, sequences):
        """
        Saves the list of generated sequences to a fasta file.
        """
        #ensure the directory exists
        os.makedirs(os.path.join(self.model_dir, 'generated', self.checkpoint_name, self.temp), exist_ok=True)

        with open(os.path.join(self.model_dir, 'generated', self.checkpoint_name, self.temp, "sequences_{}.fasta".format(self.name)), "w") as f:
            for i, seq in enumerate(sequences):
                f.write(f">{self.name}_{i}\n")
                f.write(f"{seq}\n")
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name to load.")
    parser.add_argument("--checkpoint", default="latest", type=str, help="Checkpoint name to load.")
    parser.add_argument("--text", default="all_prompts", type=str, help="Text to condition on.")
    parser.add_argument("--temp", default=0.3, type=float, help="Temperature for generation")
    parser.add_argument("--top_p", default=0.95, type=float, help="Top p for generation")
    parser.add_argument("--batch_size", default=45, type=int, help="Batch size for generation.")
    parser.add_argument("--num_seqs", default=990, type=int, help="Number of sequences to generate")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))


    with open('data/useful_from_ProteinDT/common/text_sequences.txt', 'r') as f:
        train_common_text_prompts = f.read().splitlines()
    with open('data/useful_from_ProteinDT/rare/text_sequences.txt', 'r') as f:
        train_rare_text_prompts = f.read().splitlines()

    all_prompts = train_common_text_prompts + train_rare_text_prompts


    #intialize tqdm with length EC * num_seqs/batch_size
    tqdm_length = len(all_prompts) * args.num_seqs // args.batch_size
    tqdm_iterator = tqdm(range(tqdm_length), desc="Generating sequences")

    for split, prompts in zip(['common', 'rare'], [train_common_text_prompts, train_rare_text_prompts]):
        for i, prompt in enumerate(prompts):
            prompt_name = split + '_' + str(i)
                
            #reintialize here so the seed is set correctly
            runner = Runner(model_name=args.model, checkpoint_name=args.checkpoint)

            all_sequences = []
            for batch in range(args.num_seqs // args.batch_size): #45 is the max batch size that fits on 40GB A100
                conditions = {}
                conditions['text'] = prompt

                sequences = runner.sample(name=prompt_name, conditions=conditions, temperature=args.temp, num_return_sequences=args.batch_size, top_p=args.top_p)
                all_sequences.extend(sequences)
                runner.save_seqs(all_sequences)
                # print(sequences[:10])   
                tqdm_iterator.update(1)

if __name__ == "__main__":
    main()
