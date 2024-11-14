#convert and save the output to pdb
import os
import pandas as pd
from Bio import SeqIO
import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

class ESMFold():
    def __init__(self):
        self.device = torch.device("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1", cache_dir="data/pretrained_ESMFold")
        self.model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", cache_dir="data/pretrained_ESMFold", low_cpu_mem_usage=True).to(self.device)

        #use the tips from the tutorial to speed up calculation and reduces memory usage
        self.model.esm = self.model.esm.half()
        torch.backends.cuda.matmul.allow_tf32 = True
        self.model.trunk.set_chunk_size(64)

    def get_plddt(self, protein):
        tokenized_input = self.tokenizer([protein], return_tensors="pt", add_special_tokens=False)['input_ids'].to(self.device)
        with torch.no_grad():
            output = self.model(tokenized_input)
            return torch.mean(output['plddt'].cpu()).item()
        
            # pdb = convert_outputs_to_pdb(output)
            # with open(f"structures/{ec}/{name}.pdb", "w") as f:
            #     f.write("".join(pdb))

    @staticmethod
    def convert_outputs_to_pdb(outputs):
        final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
        outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
        final_atom_positions = final_atom_positions.cpu().numpy()
        final_atom_mask = outputs["atom37_atom_exists"]
        pdbs = []
        for i in range(outputs["aatype"].shape[0]):
            aa = outputs["aatype"][i]
            pred_pos = final_atom_positions[i]
            mask = final_atom_mask[i]
            resid = outputs["residue_index"][i] + 1
            pred = OFProtein(
                aatype=aa,
                atom_positions=pred_pos,
                atom_mask=mask,
                residue_index=resid,
                b_factors=outputs["plddt"][i],
                chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
            )
            pdbs.append(to_pdb(pred))
            
        return pdbs