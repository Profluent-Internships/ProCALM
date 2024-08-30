from bioinformatics import *
import numpy as np
from tqdm import tqdm
import argparse
import torch

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

def get_accuracy_level(predicted_ECs, true_ECs):
    """
    Based on a list of predicted_ECs, calculates the highest level of accuracy achieved, against all true_ECs. Returns a list of the same length as true_ECs.
    """
    #convert true_EC to a list
    if type(predicted_ECs) == str:
        predicted_ECs = [predicted_ECs]
        
    if type(true_ECs) == str:
        true_ECs = [true_ECs]

    maxes = []
    for true_EC in true_ECs:
        
        true_split = true_EC.split('.')
        
        counters = []
        for predicted_EC in predicted_ECs:
            try:
                predicted_split = predicted_EC.split('.')
                counter = 0
    
                for predicted, true in zip(predicted_split, true_split):
                    if predicted == true:
                        counter += 1
                    else:
                        break
                counters.append(counter)
                #print(counters)
            except:
                print("ERROR:", predicted_EC)
        
        maxes.append(np.max(counters))
    return maxes

def process_lineage(lineage):
    """
    Returns the kingdom of a lineage outputted from mmseqs taxonomy.
    """
    lineage = str(lineage).split(';')
    if len(lineage) >= 2:
        return lineage[1] #second entry is the kingdom
    else:
        return float("nan")

def get_average_prediction_similarity(reference_ec, predicted_ECs, ec2encoding):
    """
    Calculates the average prediction similarity between a reference EC and a list of predicted ECs.
    The similarity is a cosine similarity using the encodings from ec2encoding.
    """
    if reference_ec == "no-ec":
        return None
    else:
        #split predicted_ECs if there is a ; character in it
        predicted_ECs = [ec for ec in predicted_ECs for ec in ec.split('; ')]
        
        predicted_ec2encoding = {ec: ec2encoding[ec] for ec in predicted_ECs if ec in ec2encoding}
        if len(predicted_ec2encoding) == 0:
            return 0
        else:
            fingerprints = torch.stack(list(predicted_ec2encoding.values()))
            return torch.cosine_similarity(ec2encoding[reference_ec], fingerprints).max().item()

def get_similarity(reference_ec, fingerprints):
    """
    Calculates the similarity between a reference EC and a set of fingerprints.
    """
    if reference_ec == "no-ec":
        return None
    else:
        return torch.cosine_similarity(ec2drfp[reference_ec], fingerprints).max().item()

def tabulate_results(summary_df, model, checkpoint, temp, ec="no-ec", tax="no-tax", split="train", low_bacteria_baseline=False):
    """
    Script to tabulate statistics on the generated sequences.
    """
    if model == 'ZymCTRL':
        file = 'results/{}/generated/sequences_pretrained_{}.fasta'.format(model, ec)
    else:
        file = 'results/{}/generated/{}/{}/sequences_{}_{}.fasta'.format(model, checkpoint, temp, ec, tax)
        
    with open(file, 'r') as f:
        lines = f.readlines()
        sequences = [clean(truncate(l, ['1', '2'])) for l in lines if not l.startswith('>')]
        n_generated = len(sequences)
        sequences = [s for s in sequences if s != ""]
        n_seqs = len(sequences)

    frac_terminated = n_seqs/n_generated
    results_df = run_bioinformatics(seqs_or_fasta=sequences, ref_db='data/ref_databases/swissprot')

    results_df.dropna(inplace=True)
    results_df['Entry'] = results_df['ref_entry_id'].apply(lambda x: x.split('|')[1])
    results_df = results_df[results_df['aln_coverage'] > 80]
    #in the future add a filter for the tantan regions with low complexity here
    
    n_good = len(results_df)
    frac_good = n_good/n_generated

    #filter to only enzyme hits
    results_df = results_df.merge(metadata, on='Entry', how='left')
    enzyme_df = results_df.dropna().reset_index()

    n_enzymes = len(enzyme_df)
    if n_good > 0:
        frac_enzymes = n_enzymes/n_good
    else:
        frac_enzymes = 0

    if ec != "no-ec":
        enzyme_df['EC numbers'] = enzyme_df['EC number'].apply(lambda x: x.split('; '))
        enzyme_df['Accuracy Level'] = enzyme_df['EC numbers'].apply(lambda x: get_accuracy_level(ec, x))
        enzyme_df['Accuracy Level'] = enzyme_df['Accuracy Level'].apply(lambda x: np.max(x))
        average_accuracy_level = enzyme_df['Accuracy Level'].mean()
        generated_ECs = enzyme_df['EC number'].values.tolist()

        correct_ec_df = enzyme_df[enzyme_df['EC number'] == ec].reset_index()
        n_ec_correct = len(correct_ec_df)
        if n_enzymes != 0:
            frac_ec_correct = n_ec_correct/n_enzymes
        else:
            frac_ec_correct = 0
        
        if ec in train_dist['ec'].keys():
            ec_enrichment = frac_ec_correct/train_dist['ec'][ec]
        else:
            if frac_ec_correct == 0:
                ec_enrichment = 0
            else:
                ec_enrichment = np.inf
    else:
        correct_ec_df = enzyme_df
        average_accuracy_level = None
        generated_ECs = None
        n_ec_correct = None
        frac_ec_correct = None
        ec_enrichment = None

    if tax != "no-tax":

        superkingdom = '2' if low_bacteria_baseline == True else tax.split('.')[0]

        save_path = f"results/{model}/mmseqs_taxonomy/{ec}_{tax}"
        #if save path already exists, delete it and reinitailize as empty
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)

        #save the good sequences to a new fasta file
        sequences = enzyme_df['sequence'].values
        with open(f'{save_path}/processed_sequences.fasta', 'w') as f:
            for i, sequence in enumerate(sequences):
                f.write(f'>seq{i}\n')
                f.write(f'{sequence}\n')

        os.system(f"{download_mmseqs()} easy-taxonomy {save_path}/processed_sequences.fasta data/ref_databases/mmseqs/swissprot {save_path}/alnRes data/ref_databases/mmseqs/tmp --lca-ranks superkingdom,phylum,class,order,family,genus,species --tax-lineage 2 -v 1")

        if os.path.exists(f'{save_path}/alnRes_lca.tsv'):
            tax_results_df = pd.read_csv(f'{save_path}/alnRes_lca.tsv', sep='\t', header=None)
            tax_results_df['superkingdom'] = tax_results_df[5].apply(process_lineage)

            frac_tax_mapped = len(tax_results_df.dropna())/n_enzymes

            correct_tax_results_df = tax_results_df[tax_results_df['superkingdom'] == superkingdom]
            n_tax_correct = len(correct_tax_results_df)
            correct_indices = correct_tax_results_df.index.values
           
            # frac_tax_correct = n_tax_correct/len(tax_results_df.dropna()) #this may have inflated performance
            frac_tax_correct = n_tax_correct/n_enzymes #this one is a conservative estimate
            tax_enrichment = frac_tax_correct/train_dist['tax'][superkingdom]
        
            both_correct_df = correct_ec_df[correct_ec_df.index.isin(correct_indices)]
        else:
            frac_tax_mapped = None
            frac_tax_correct = None
            n_tax_correct = None
            tax_enrichment = None
            both_correct_df = pd.DataFrame()
    else:
        both_correct_df = correct_ec_df
        n_tax_correct = None
        frac_tax_mapped = None
        frac_tax_correct = None
        tax_enrichment = None
    
    n_both_correct = len(both_correct_df)
    frac_both_correct = n_both_correct/n_enzymes if n_enzymes > 0 else None

    if ec != "no-ec" and tax != "no-tax":
        both_counts = len(train_df[(train_df['EC number'] == ec) & (train_df['superkingdom'] == superkingdom)])
        train_frac = both_counts/len(train_df)
        if train_frac > 0 and frac_both_correct is not None:
            both_enrichment = frac_both_correct/train_frac 
        else:
            both_enrichment = None
    else:
        both_counts = None
        both_enrichment = None

    #calculate the clusters in the enzyme_df (used to be both_correct_df)
    avg_max_id = enzyme_df['max_id'].mean()/100 if n_enzymes > 0 else None 
    frac70_clusters = enzyme_df['cluster_70'].nunique()/n_enzymes if n_enzymes > 0 else None 
    frac90_clusters = enzyme_df['cluster_90'].nunique()/n_enzymes if n_enzymes > 0 else None 

    summary_df.loc[len(summary_df.index)] = [model, checkpoint, ec, tax, split, n_generated, frac_terminated, frac_good, n_good, frac_enzymes, n_enzymes, average_accuracy_level, frac_ec_correct, n_ec_correct, ec_enrichment, frac_tax_mapped, frac_tax_correct, n_tax_correct, tax_enrichment, frac_both_correct, both_enrichment, avg_max_id, frac70_clusters, frac90_clusters, generated_ECs, both_counts]
    return summary_df

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name to load.")
    parser.add_argument("--checkpoint", default="pretrained", type=str, help="Checkpoint name to load.")
    parser.add_argument("--ec", default=None, type=str, help="EC number to conditionally generate from. train+test specifies a list of curated ECs.")
    parser.add_argument("--tax", default=None, type=str, help="Taxonomy lineage IDS to conditionally generate from")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    os.chdir('../')

    metadata = pd.read_csv("data/ref_databases/swissprot_enzyme.tsv", sep='\t')
    train_df = pd.read_csv('data/CARE_resampled50cluster_medium_withTax/train.csv')
    test_df = pd.read_csv('data/CARE_resampled50cluster_medium_withTax/test.csv')
    all_df = pd.concat([train_df, test_df], axis=0)

    train_dist = {}
    train_dist['ec'] = dict(train_df['EC number'].value_counts()/len(train_df))
    train_df['superkingdom'] = train_df['Tax number'].apply(lambda x: x.split('.')[0]) 
    train_dist['tax'] = dict(train_df['superkingdom'].value_counts().sort_index()/len(train_df))

    #load as a list from a txt file
    with open('data/ECs_generation/train_common_ecs.txt', 'r') as f:
        train_common_ecs = f.read().splitlines()
    with open('data/ECs_generation/train_rare_ecs.txt', 'r') as f:
        train_rare_ecs = f.read().splitlines()
    with open('data/ECs_generation/test_ecs.txt', 'r') as f:
        test_ecs = f.read().splitlines()
    with open('data/ECs_generation/low_bacteria_common_ecs.txt', 'r') as f:
        low_bacteria_ecs = f.read().splitlines()

    models = [args.model]
    checkpoints = [args.checkpoint] 
    temps = ["temp0.3"]

    if args.ec == 'train+test':
        ecs = train_common_ecs + train_rare_ecs + test_ecs
    elif args.ec == 'rare':
        ecs = train_rare_ecs    
    elif args.ec == 'common':
        ecs = train_common_ecs  
    elif args.ec == "low_bacteria":
        ecs = low_bacteria_ecs 
    elif args.ec == None:
        ecs = ["no-ec"]
    else:
        ecs = [args.ec]
    
    if args.tax == 'superkingdoms':
        taxes = ['2.1224.1236.2887326.468.469.470', '2157.2283796.183967.2301.46630.46631.46632', '2759.4890.147550.5125.5129.5543.51453', '10239.2731618.2731619.-1.2946170.10663.-1']
    elif args.tax == None:
        taxes = ['no-tax']
    else:
        taxes = [args.tax]

    #check to make sure all files exist before proceedding
    flag = False
    for model in models:
        for checkpoint in checkpoints:
            for temp in temps:
                if model == 'ZymCTRL':
                    ecs = train_common_ecs + train_rare_ecs

                for ec in ecs:
                    for tax in taxes:
                        if model == 'ZymCTRL':
                            file = 'results/{}/generated/sequences_pretrained_{}.fasta'.format(model, ec)
                            temp = 'temp1'
                        else:
                            file = 'results/{}/generated/{}/{}/sequences_{}_{}.fasta'.format(model, checkpoint, temp, ec, tax)
                        if not os.path.exists(file):
                            print(file + " does not exist")
                            flag = True
    if flag:
        exit()

    pbar = tqdm(total=len(models) * len(checkpoints) * len(temps) * len(ecs) * len(taxes), desc='Processing')

    for model in models:
        low_bacteria_basline = True if "lowbacteria" in model else False

        for checkpoint in checkpoints:
            for temp in temps:
                summary_df = pd.DataFrame(columns=['model', 'checkpoint', 'ec', 'tax', 'split', 'n_generated', 'frac_terminated', 'frac_good', 'n_good', 'frac_enzymes', 'n_enzymes', 'average_accuracy_level', 'frac_ec_correct', 'n_ec_correct', 'ec_enrichment', 'frac_tax_mapped', 'frac_tax_correct', 'n_tax_correct', 'tax_enrichment', 'frac_both_correct', 'both_enrichment', 'avg_max_id',  'frac_70clusters', 'frac_90clusters', 'generated_ECs', "both_count"])

                for ec in ecs:
                    if ec in train_common_ecs:
                        split = 'train_common'
                    elif ec in train_rare_ecs:
                        split = 'train_rare'
                    elif ec in test_ecs:
                        split = 'test'
                    else:
                        split = None
                    
                    for tax in taxes:
                        pbar.set_postfix(ec=ec, tax=tax)
                        summary_df = tabulate_results(summary_df, model, checkpoint, temp, ec=ec, tax=tax, split=split, low_bacteria_baseline=low_bacteria_basline)
                        pbar.update(1)

                all_ec_value_counts = all_df['EC number'].value_counts()
                summary_df = summary_df.merge(all_ec_value_counts, left_on='ec', right_index=True, how='left')
                summary_df.rename(columns={'count': 'ec_count'}, inplace=True)

                all_tax_value_counts = train_df['superkingdom'].value_counts()
                summary_df = summary_df.merge(all_tax_value_counts, left_on='tax', right_index=True, how='left')
                summary_df.rename(columns={'count': 'tax_count'}, inplace=True)

                ec2drfp = torch.load('data/ec2drfp.pt')

                if ec != "no-ec":
                    summary_df['average_prediction_similarity'] = summary_df.apply(lambda x: get_average_prediction_similarity(x['ec'], x['generated_ECs'], ec2drfp), axis=1)

                    #get similarity to the training set
                    train_ecs = train_df['EC number'].unique()
                    train_ec2drfp = {ec: ec2drfp[ec] for ec in train_ecs}
                    fingerprints = torch.stack(list(train_ec2drfp.values()))
                    summary_df['similarity_to_train'] = summary_df['ec'].apply(get_similarity, args=(fingerprints,))

                #load the summary_df if it already exists and append new results
                if os.path.exists(f'results/{model}/all_summary_{checkpoint}_{temp}.csv'):
                    old_summary_df = pd.read_csv(f'results/{model}/all_summary_{checkpoint}_{temp}.csv')
                    summary_df = pd.concat([old_summary_df, summary_df], axis=0)

                summary_df.to_csv('results/{}/all_summary_{}_{}.csv'.format(model, checkpoint, temp), index=False)