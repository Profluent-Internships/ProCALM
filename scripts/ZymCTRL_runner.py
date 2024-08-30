import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
import os
from tqdm import tqdm
import math

### This script is used to generate sequences from the ZymCTRL model and is modified from https://huggingface.co/AI4PD/ZymCTRL.

def remove_characters(sequence, char_list):
    "This function removes special tokens used during training."
    columns = sequence.split('<sep>')
    seq = columns[1]
    for char in char_list:
        seq = seq.replace(char, '')
    return seq

def calculatePerplexity(input_ids,model,tokenizer):
    "This function computes perplexities for the generated sequences"
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)
        
def main(label, model,special_tokens,device,tokenizer):
    # Generating sequences
    input_ids = tokenizer.encode(label,return_tensors='pt').to(device)
    outputs = model.generate(
        input_ids, 
        top_k=9, #tbd
        repetition_penalty=1.2,
        max_length=1024,
        eos_token_id=1,
        pad_token_id=0,
           do_sample=True,
           num_return_sequences=20) # Depending non your GPU, you'll be able to generate fewer or more sequences. This runs in a 40GB A100.
    
    # Check sequence sanity, ensure sequences are not-truncated.
    # The model will truncate sequences longer than the specified max_length (1024 above). We want to avoid those sequences.
    # new_outputs = [ output for output in outputs if output[-1] == 0]
    # if not new_outputs:
    #     print("not enough sequences with short lengths!!")
    #skip this for now
    new_outputs = outputs

    # Compute perplexity for every generated sequence in the batch
    ppls = [(tokenizer.decode(output), calculatePerplexity(output, model, tokenizer)) for output in new_outputs ]

    # Sort the batch by perplexity, the lower the better
    ppls.sort(key=lambda i:i[1]) # duplicated sequences?

    # Final dictionary with the results
    sequences={}
    sequences[label] = [(remove_characters(x[0], special_tokens), x[1]) for x in ppls]

    return sequences

if __name__=='__main__':
    #set seed
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True

    os.chdir('../')

    device = torch.device("cuda") 
    print('Reading pretrained model and tokenizer')
    tokenizer = AutoTokenizer.from_pretrained('AI4PD/ZymCTRL', cache_dir="data/pretrained_ZymCTRL")
    model = GPT2LMHeadModel.from_pretrained('AI4PD/ZymCTRL', cache_dir="data/pretrained_ZymCTRL").to(device)
    special_tokens = ['<start>', '<end>', '<|endoftext|>','<pad>',' ', '<sep>']

    labels = ["4.2.1.20"]

    for label in tqdm(labels):
        # We'll run 45 batches per label. 20 sequences will be generated per batch seems to fit on 40GB A100.
        sequences_raw = []
        for i in range(0,45): 
            sequences = main(label, model, special_tokens, device, tokenizer)
            for key, value in sequences.items():
                for index, val in enumerate(value):
                    sequences_raw.append(val[0])
        
        #write to fasta
        directory = "results/ZymCTRL/generated"
        os.makedirs(os.path.dirname(directory), exist_ok=True)
        filename =  directory + "/sequences_pretrained_{}.fasta".format(label)
        with open(filename, "w") as fn:
            for j, seq in enumerate(sequences_raw):
                fn.write(f'>{j}\n')
                fn.write(f'{seq}\n')