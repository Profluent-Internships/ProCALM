# ProCALM
ProCALM (Protein Conditionally Adapted Language Model) is a method where Progen2-base is finetuned with conditional adapters for conditional generation of functional enzymes, based on EC number, taxonomy, or both. Model here is refered to as progen-conditional. This is the version of the repo cleared for external use.

## Setup and Installation
 We have provided `docker/Dockerfile` to build this image. Alternatively, model will run in a conda environment created using `docker/environment.yml`. All results can be downloaded from [here]() and unzipped to replace the `results/checkpoints` folder.

## Dataset Processing
For convenience, useful and publicly available datasets from [CARE benchmarks](https://github.com/jsunn-y/CARE/) are preloaded in this repo. To reproduce the dataset processing steps, the raw data used for training ProCALM is obtained from [Uniprot](https://www.uniprot.org). You will need all enzymes with EC numbers and taxonomy lineages from uniprot for Uniref (>20M) and Swissprot (~200k). Use the versions from June 17, 2024 and export EC number, sequence, entry ID, taxonomy, taxonomy ID, lineage, lineage ID as a tsv. Raw data is alternatively uploaded [here]() and can fill in the `data/raw_data` folder. For building the Diamond reference database, you will also need swissprot.fasta file (~550k sequences from June 17, 2024 download). Dataset processing and sequence analysis requires mmseqs, diamond blast and tantan. Diamond reference databases along with mmseqs databases including taxonomy need to built before analysis using `database_setup.py`. This script will also run sequence clustering on swissprot. 

The final splits used in our study are built and saved to sharded datasets using `build_splits.ipynb` under `scripts`. `save_sharded.ipynb` is a shortcut to save sharded datasets from presaved csv files of each split (in particular the sharded version of uniref). You can also use `select_ECs.ipynb` to reproduce how we generated the EC numbers that are used for generation and analysis. 

## Training
Example command for running on 4 40GB A100s:
```
composer main.py --config config/long-final/ec-onehot-swissprot.yml --debug
```
Results will be saved under `results/`. Training should take on the order of 6 hours for every 1 billion tokens.

## Generation
Example commands:
```
#generate for a single EC conditioning
python runner.py --model ec-onehot-swissprot_20240819-231400 --checkpoint ba21000 --ec 4.2.1.20

#generate for EC conditionings tested in our study
python runner.py --model ec-onehot-swissprot_20240819-231400 --checkpoint ba21000 --ec train+test --num_seqs 225

#generate for EC and taxonomy conditioning
python runner.py --model ec+tax-swissprot_20240819-231401 --checkpoint ba21000 --ec common --tax 2.1224.1236.2887326.468.469.470 --num_seqs 225

#generate for ECs with few bacteria
python runner.py --model ec+tax-swissprot-lowbacteria_20240824-202511 --checkpoint ba11000 --ec low_bacteria --tax superkingdoms

#ZymCTRL baseline for rare and common ECs
python ZymCTRL_runner.py
```
Alternatively, if 4 GPUs are available, you can run many generations in parallel with `scripts/parallel_generation/run_parallel_generation.sh`. Outputs will be saved under `results/{model_name}/generated/{checkpoint}`. Note that parallelization should allow the generation to finish in under 2 hours if you generating with 225 sequences per EC and 72 unique ECs. Note that some of the generation may come out slightly differently, as there is an update with seeding added at a different point in the runner.

## Analysis
Here are example commands to reproduce our study and get statistics on the generation quality and diversity of generated sequences under `scripts`:
```
#ZymCTRL Baseline
python run_generation_processing.py --model ZymCTRL --ec train+test

#our model
python run_generation_processing.py --model ec-onehot-swissprot_20240819-231400 --checkpoint ba21000 --ec train+test

#taxonomy conditioning
python run_generation_processing.py --model tax-swissprot_20240820-004651 --checkpoint ba63000 --tax superkingdoms

#joint conditioning
python run_generation_processing.py --model ec+tax-swissprot_20240819-231401 --checkpoint ba21000 --ec common --tax superkingdoms

#ECs with few bacteria
python run_generation_processing.py --model ec+tax-swissprot-lowbacteria_20240824-202511 --checkpoint ba11000 --ec low_bacteria --tax superkingdoms
```
Results will be saved under `results/{model_name}/all_summary_{checkpoint}.csv`. Analysis will fun faster on more CPUs but should take on the order of seconds to one minute per EC/tax.

Perplexities for different splits can be calculated with `perplexity_calculation.py` under `scripts` and outputs will be saved under `results/{model_name}/perplexity_{checkpoint}.csv`. Perplexity calculation requires a single A100 and should take on the order of minutes to one hour for all datasets on a single checkpoint.

The output csvs can be visualized using `analysis/visualization.ipynb`.

### Limitations
Some of the results did not make it into the final figures (and their respective deprecated models have not been saved). For example, using separate projectors/encoders for every adapter layer does not seem to lead to improved performance. Bottlenecking the low rank projection of the LM hidden embedding also leads to negligible improvements to conditional generation performance.

Future work could include using the uniref dataset for all model finetuning and also more detailed analysis of the generated sequences using CLEAN or ESMFold.

