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
