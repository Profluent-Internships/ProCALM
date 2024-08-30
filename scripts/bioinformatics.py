import os
import pathlib
import shutil
import tarfile
import tempfile
import urllib
from typing import List, Union

import Bio.SeqIO
import pandas as pd


def download_diamond() -> str:
    cache_dir = os.path.join(str(pathlib.Path.home()), ".cache")
    if not os.path.exists(os.path.join(cache_dir, "diamond")):
        fname = os.path.join(cache_dir, "diamond.tar.gz")
        urllib.request.urlretrieve(
            "https://github.com/bbuchfink/diamond/releases/download/v2.1.9/diamond-linux64.tar.gz", fname
        )  # nosec
        with tarfile.open(fname, "r:gz") as f:
            f.extractall(cache_dir)  # nosec
        os.remove(fname)
    return os.path.join(cache_dir, "diamond")


def download_mmseqs() -> str:
    cache_dir = os.path.join(str(pathlib.Path.home()), ".cache")
    if not os.path.exists(os.path.join(cache_dir, "mmseqs")):
        fname = os.path.join(cache_dir, "mmseqs.tar.gz")
        urllib.request.urlretrieve(
            "https://github.com/soedinglab/MMseqs2/releases/download/15-6f452/mmseqs-linux-avx2.tar.gz", fname
        )  # nosec
        with tarfile.open(fname, "r:gz") as f:
            f.extractall(cache_dir)  # nosec
        os.remove(fname)
    return os.path.join(cache_dir, "mmseqs", "bin", "mmseqs")


def download_tantan() -> str:
    if not shutil.which("tantan"):
        os.system("apt-get update -y && apt-get install tantan")  # nosec
    return "tantan"


def seqs_to_fasta(seqs: List[str], fname: str):
    with open(fname, "w") as f:
        for i, seq in enumerate(seqs):
            f.write(f">{i}\n{seq}\n\n")


def make_diamond_db(seqs_or_fasta: Union[str, List[str]], diamond_fname: str) -> None:
    if isinstance(seqs_or_fasta, str):
        fasta_fname = seqs_or_fasta
        f = None
    else:
        f = tempfile.NamedTemporaryFile("w")
        fasta_fname = f.name
        seqs_to_fasta(seqs_or_fasta, fasta_fname)
    os.system(f"{download_diamond()} makedb --in {fasta_fname} --db {diamond_fname} --ignore-warnings")  # nosec
    if f is not None:
        f.close()


def diamond_alignment(seqs_or_fasta: Union[str, List[str]], ref_db: str) -> pd.DataFrame:
    temp_dir = tempfile.mkdtemp()
    if isinstance(seqs_or_fasta, str):
        fasta = seqs_or_fasta
        seqs = {str(s.id): str(s.seq) for s in Bio.SeqIO.parse(fasta, "fasta")}
    else:
        fasta = os.path.join(temp_dir, "in.fasta")
        seqs = {str(i): seq for i, seq in enumerate(seqs_or_fasta)}
        seqs_to_fasta(seqs_or_fasta, fasta)
    diamond = download_diamond()
    output = os.path.join(temp_dir, "diamond.tsv")
    outcols = "qseqid scovhsp pident sseqid"
    os.system(f"{diamond} blastp -k 1 --query {fasta} --db {ref_db} --out {output} --outfmt 6 {outcols} --quiet")  # nosec
    
    if not os.path.exists(output):
        df = pd.DataFrame(columns=["sequence", "aln_coverage", "max_id", "ref_entry_id"])
    else:
        df = pd.read_csv(output, sep="\t", header=None, index_col=0, names=["aln_coverage", "max_id", "ref_entry_id"])
        if len(df) == 0:
            df = pd.DataFrame(columns=["sequence", "aln_coverage", "max_id", "ref_entry_id"])
        else:
            df["sequence"] = [seqs[i] for i in df.index.astype(str)]
    
    df["max_id"] *= df["aln_coverage"] / 100
    return df.reset_index(drop=True).drop_duplicates("sequence")

def run_tantan(seqs_or_fasta: Union[str, List[str]]) -> pd.DataFrame:
    if isinstance(seqs_or_fasta, str):
        fasta_fname = seqs_or_fasta
        f = None
    else:
        f = tempfile.NamedTemporaryFile("w")
        fasta_fname = f.name
        seqs_to_fasta(seqs_or_fasta, fasta_fname)
    f2 = tempfile.NamedTemporaryFile("w")
    os.system(f"{download_tantan()} -p {fasta_fname} > {f2.name}")  # nosec
    seqs = [str(s.seq) for s in Bio.SeqIO.parse(f2.name, "fasta")]
    f2.close()
    if f is not None:
        f.close()
    df = pd.DataFrame({"sequence": [s.upper() for s in seqs]})
    df["n_low_complexity"] = [sum(a.islower() for a in s) for s in seqs]
    return df

def cluster(seqs_or_fasta: Union[str, List[str]], seq_id: float):
    temp_dir = tempfile.mkdtemp().rstrip("/") + "/"
    if isinstance(seqs_or_fasta, str):
        fasta = seqs_or_fasta
    else:
        fasta = os.path.join(temp_dir, "in.fasta")
        seqs_to_fasta(seqs_or_fasta, fasta)
    seqs = {str(s.id): str(s.seq) for s in Bio.SeqIO.parse(fasta, "fasta")}
    tmp = os.path.join(temp_dir, "tmp")
    os.system(f"{download_mmseqs()} easy-cluster {fasta} {temp_dir} {tmp} --min-seq-id {seq_id} -v 1")  
    # if file doesnt exist, skip
    if not os.path.exists(os.path.join(temp_dir, "_cluster.tsv")):
        return pd.DataFrame(columns=["sequence", f"cluster_{int(seq_id * 100)}"])
    else:
        df = pd.read_csv(os.path.join(temp_dir, "_cluster.tsv"), sep="\t", index_col=1, header=None)
        df.columns = [f"cluster_{int(seq_id * 100)}"]
        df["sequence"] = [seqs[str(i)] for i in df.index]
    return df.iloc[:, ::-1].drop_duplicates("sequence")

def run_bioinformatics(
    seqs_or_fasta: Union[str, List[str]], ref_db: str = None, cluster_ids=(0.5, 0.7, 0.9)
) -> pd.DataFrame:
    if isinstance(seqs_or_fasta, str):
        fasta = seqs_or_fasta
        f = None
    else:
        f = tempfile.NamedTemporaryFile("w")
        fasta = f.name
        seqs_to_fasta(seqs_or_fasta, fasta)
    df = run_tantan(fasta)
    if ref_db is not None:
        new_df = diamond_alignment(fasta, ref_db)
        if len(df) == 0 and len(new_df) == 0:
            df = pd.DataFrame(columns=["sequence", "n_low_complexity", "aln_coverage", "max_id", "ref_entry_id"])
        else:
            df = df.merge(new_df, on="sequence", how="left")
    for cluster_id in cluster_ids:
        if df.empty:
            df = pd.DataFrame(columns=["sequence", "n_low_complexity", "aln_coverage", "max_id", "ref_entry_id", f"cluster_{int(cluster_id * 100)}"])
        else:
            df = df.merge(cluster(fasta, cluster_id), on="sequence")
    if f is not None:
        f.close()
    return df
