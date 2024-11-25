import os
import pandas as pd

#before running, install foldseek with
#conda install -c conda-forge -c bioconda foldseek
model =  'ec-onehot-swissprot' #'ZymCTRL'
checkpoint =  'ba11000' #'pretrained'

with open('../../data/ECs_generation/train_common_ecs.txt') as f:
    ec_list = f.read().splitlines()

fractions_unique_clusters = []
nunique_clusters = []

#drop '4.2.99.18'
for ec in ec_list:
    #check if the folder exists
    if os.path.exists(f"../../results/{model}/generated/{checkpoint}/temp0.3/{ec}"):

        os.system(f'foldseek easy-cluster ../../results/{model}/generated/{checkpoint}/temp0.3/{ec} ../../results/{model}/generated/{checkpoint}/temp0.3/{ec}/foldseek_res  ../../tmp -c 0.9')

        cluster_df = pd.read_csv(f"../../results/{model}/generated/{checkpoint}/temp0.3/{ec}/foldseek_res_cluster.tsv", sep='\t', header=None)

        nunique = cluster_df[0].nunique()
        fraction_unique_clusters = nunique/len(cluster_df)
        nunique_clusters.append(nunique)
        fractions_unique_clusters.append(fraction_unique_clusters)
    else:
        nunique_clusters.append(0)
        fractions_unique_clusters.append(0)

clusters_summary_df = pd.DataFrame({'ec': ec_list, 'nunique_clusters': nunique_clusters, 'fraction_unique_clusters': fractions_unique_clusters})

clusters_summary_df.to_csv(f'../../results/{model}/generated/{checkpoint}/temp0.3/foldseek_clusters_summary.csv', index=False)

