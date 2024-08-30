import os
import pandas as pd
import tempfile
from bioinformatics import make_diamond_db, download_mmseqs, seqs_to_fasta, cluster

os.chdir('../')

### Set up the databases for the project and perform clustering ###

#ensure the directory exists
os.makedirs('data/ref_databases', exist_ok=True)

#make diamond reference database for BLAST
make_diamond_db('data/raw_data/swissprot.fasta', 'data/ref_databases/swissprot') 

#download swissprot database as an mmseqs database
os.system(f"{download_mmseqs()} databases UniProtKB/Swiss-Prot data/ref_databases/mmseqs/swissprot data/ref_databases/mmseqs/tmp")

#build mmseqs taxonomy database
os.system(f"{download_mmseqs()} createtaxdb data/ref_databases/mmseqs/swissprot data/ref_databases/mmseqs/tmp")

#create index for mmseqs
os.system(f"{download_mmseqs()} createindex data/ref_databases/mmseqs/swissprot data/ref_databases/mmseqs/tmp")

#run clustering on swissprot
df = pd.read_csv('data/raw_data/swissprot_enzyme.tsv', sep='\t')

sequences = df['Sequence'].values
if isinstance(sequences, str):
    fasta = sequences
    f = None
else:
    f = tempfile.NamedTemporaryFile("w")
    fasta = f.name
    seqs_to_fasta(sequences, fasta)

for cluster_id in (0.5, 0.7, 0.9):
    df = df.merge(cluster(fasta, cluster_id), left_on="Sequence", right_on="sequence", how="left")

df.drop(columns=["sequence", "sequence_x", "sequence_y"], inplace=True)

df.to_csv("data/processed_data/swissprot_enzyme_clustered.tsv", sep='\t', index=False)