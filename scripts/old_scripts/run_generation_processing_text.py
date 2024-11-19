from bioinformatics import *
import numpy as np
from tqdm import tqdm
import argparse
import torch
from ESMFold import ESMFold

def clean(sample):
    """
    Removes short samples and those without the proper start and end tokens.
    """
    if len(sample) < 50:
        return ""
    else:
        if sample[-1] == '2':
            sample = sample[1:-1]
            if ('1' in sample) or ('2' in sample):
                return ""
            else:
                return sample
        elif ('1' in sample) or ('2' in sample):
            return ""
        else:
            return sample

def truncate(sample, terminals):
    """
    Truncates a sequence between the correct start and end tokens.
    """
    pos = []
    for terminal in terminals:
        find_pos = sample.find(terminal, 1)
        if find_pos != -1:
            pos.append(find_pos)
    if len(pos) > 0:
        return sample[:(min(pos)+1)]
    else:
        return sample

def tabulate_results(summary_df, model, checkpoint, temp, prompt_name, prompt, plddt=False):
    """
    Script to tabulate statistics on the generated sequences.
    """
    split_name = prompt_name.split('_')[0]

    if model == 'ProteinDT':
        #for now hardcode it to 225 generated sequences
        lines_start = int(prompt_name.split('_')[1])*225*2
        lines_end = lines_start + 225*2

        with open(f'results/ProteinDT/generated/step_02_inference_{split_name}.txt', 'r') as f:
            lines = f.readlines()[lines_start:lines_end]
            #read every other line
            sequences = [l.strip()for l in lines[1::2]]
            n_generated = len(sequences)
            n_seqs = len(sequences)
            # print(sequences)
            # print(n_seqs)
    else:
        file = 'results/{}/generated/{}/{}/sequences_{}.fasta'.format(model, checkpoint, temp, prompt_name)
        
        with open(file, 'r') as f:
            lines = f.readlines()
            sequences = [clean(truncate(l, ['1', '2'])) for l in lines if not l.startswith('>')]
            n_generated = len(sequences)
            sequences = [s for s in sequences if s != ""]
            n_seqs = len(sequences)

    frac_terminated = n_seqs/n_generated
    results_df = run_bioinformatics(seqs_or_fasta=sequences, ref_db='data/ref_databases/swissprot_ProteinDT')
    results_df.dropna(inplace=True)

    results_df['index'] = results_df['ref_entry_id'].str.split('_').str[1] #remove seq_ from the ront of the name
    results_df = results_df[results_df['aln_coverage'] > 80]
    #in the future add a filter for the tantan regions with low complexity here
    
    n_good = len(results_df)
    frac_good = n_good/n_generated

    #filter to only enzyme hits
    results_df = results_df.merge(metadata[['index', 'Text']], on='index', how='left')
    #results_df = results_df.dropna().reset_index()
    #n_enzymes = len(results_df)

    #check how many of the retrieved prompts match the reference target prompt
    results_df['corect'] = results_df['Text'] == prompt
    n_correct = results_df['corect'].sum()
    frac_correct = n_correct/n_good

    correct_df = results_df[results_df['corect'] == True]
    #calculate the plddt of the correctly conditioned sequences
    if plddt and n_correct > 0:
        #to speed things up, only take statistics on up to the first 100 sequences
        both_correct_seqs = correct_df['sequence'].values[:100]
        pbar2 = tqdm(total=len(both_correct_seqs), desc='Folding')
        plddts = []
        esmfold = ESMFold()
        for seq in both_correct_seqs:
            plddts.append(esmfold.get_plddt(seq))
            pbar2.update(1)
        avg_plddt = np.mean(plddts)
    else:
        avg_plddt = None

    #calculate the clusters in the valid proteins df (used to be both_correct_df)
    avg_max_id = results_df['max_id'].mean()/100 if n_good > 0 else None 
    frac70_clusters = results_df['cluster_70'].nunique()/n_good if n_good > 0 else None 
    frac90_clusters = results_df['cluster_90'].nunique()/n_good if n_good > 0 else None 

    summary_df.loc[len(summary_df.index)] = [model, checkpoint, prompt_name, split_name, n_generated, frac_terminated, frac_good, n_good, frac_correct, n_correct, avg_max_id, frac70_clusters, frac90_clusters, avg_plddt]

    return summary_df

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name to load.")
    parser.add_argument("--checkpoint", default="pretrained", type=str, help="Checkpoint name to load.")
    parser.add_argument("--text", default="all_prompts", type=str, help="Text to condition on.")
    parser.add_argument("--plddt", action='store_true', help="Run plddt evluation on the generated sequences.")
    parser.set_defaults(plddt=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # os.chdir('../../')
    # print(os.getcwd())

    metadata = pd.read_csv("data/ref_databases/swissprot_proteinDT_text.csv")
    metadata['index'] = metadata.index.values.astype(str)

    train_dist = {}
    
    with open('data/useful_from_ProteinDT/common/text_sequence.txt', 'r') as f:
        train_common_text_prompts = f.read().splitlines()
    with open('data/useful_from_ProteinDT/rare/text_sequence.txt', 'r') as f:
        train_rare_text_prompts = f.read().splitlines()


    models = [args.model]
    checkpoints = [args.checkpoint] 
    temps = ["temp1.0"] #for text we are typically evaluating the higher temp

    if args.text == 'all_prompts':
        prompts = train_common_text_prompts + train_rare_text_prompts

    #check to make sure all files exist before proceedding
    # flag = False
    # for model in models:
    #     for checkpoint in checkpoints:
    #         for temp in temps:

    #             for split, prompts in zip(['common', 'rare'], [train_common_text_prompts, train_rare_text_prompts]):
    #                 for i, prompt in enumerate(prompts):

    #                     prompt_name = split + '_' + str(i)
    #                     file = 'results/{}/generated/{}/{}/sequences_{}.fasta'.format(model, checkpoint, temp, prompt_name)
                        
    #                     if not os.path.exists(file):
    #                         print(file + " does not exist")
    #                         flag = True
    # if flag:
    #     exit()

    pbar = tqdm(total=len(models) * len(checkpoints) * len(temps) * len(prompts), desc='Processing')

    for model in models:
        for checkpoint in checkpoints:
            for temp in temps:
                summary_df = pd.DataFrame(columns=['model', 'checkpoint', 'prompt_name', 'split', 'n_generated', 'frac_terminated', 'frac_good', 'n_good', 'frac_correct', 'n_correct', 'avg_max_id',  'frac_70clusters', 'frac_90clusters', "avg_plddt"])
                
                for split, prompts in zip(['common', 'rare'], [train_common_text_prompts, train_rare_text_prompts]):
                    for i, prompt in enumerate(prompts):
                        prompt_name = split + '_' + str(i)

                        summary_df = tabulate_results(summary_df, model, checkpoint, temp, prompt_name=prompt_name, prompt=prompt, plddt=args.plddt)
                        pbar.update(1)

                #load the summary_df if it already exists and append new results
                if os.path.exists(f'results/{model}/all_summary_{checkpoint}_{temp}.csv'):
                    old_summary_df = pd.read_csv(f'results/{model}/all_summary_{checkpoint}_{temp}.csv')
                    summary_df = pd.concat([old_summary_df, summary_df], axis=0)

                summary_df.to_csv('results/{}/all_summary_{}_{}.csv'.format(model, checkpoint, temp), index=False)