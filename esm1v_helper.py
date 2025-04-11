# Standard modules
import glob

# Third-party modules
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

def __main__():
    
    """
    --- USER CONFIG ---
    Please fill in the file paths below:
    - Fasta file (str) - A fasta file containing the WT sequences and their names
    - Mutation folder path (str) - A folder containing the files of mutants and their WT names
    - Batch size (int) - Number of sequences to process at once, larger number is faster but requires more memory
    - Results path (str) - The path of the file you would like to save your results in, if this already exists it will be overwritten
    """
    
    fasta_file_path = "/path/to/fasta_file.fs"
    mutation_folder_path = "/path/to/mutations_folder"
    batch_size = 100
    results_path = "/path/to/results_file.tsv"
    
    
    
    sequences_dict = read_fasta_file(fasta_file_path)
    sequences_df = pd.DataFrame(list(sequences_dict.items()), columns=["sequence_name", "wt_sequence"])
    
    mutation_files = glob.glob(mutation_folder_path + "/*.txt")
    mutation_dfs = []
    
    for file in mutation_files:
        
        temp_df = pd.read_csv(file, sep = " ", header = None, names = ["sequence_name", "mutation"])
        mutation_dfs.append(temp_df)
    
    model, tokeniser, device = setup_model()
    mutations_df = pd.concat(mutation_dfs, ignore_index=True)
    merged_df = mutations_df.merge(sequences_df, on = "sequence_name", how = "left")
    results_df = get_mutation_effect_score(merged_df, model, tokeniser, device, batch_size)
    
    print(results_df)
    
    results_df.to_csv("")
    
def read_fasta_file(fasta_file: str) -> dict:
    
    """
    Reads each line of the indicated fasta file, sorting the sequences into a dictionary keyed by name
    
    Inputs:
    - Fasta file (str): The path to the fasta file of WT sequences
    
    Outputs:
    - Sequences (dict): Dictionary containing each WT sequence, keyed by its given name
    """
    
    sequences = {}
    current_sequence_name = None
    sequence_lines = []
    
    with open(fasta_file, "r") as f:
        
        for line in f:
            
            line = line.strip()
            
            if line.startswith(">"):
                
                if current_sequence_name is not None:
                    
                    sequences[current_sequence_name] = "".join(sequence_lines)
                    
                current_sequence_name = line[1:]
                sequence_lines = []
                
            else:
                
                sequence_lines.append(line)
                
        if current_sequence_name is not None:
            
            sequences[current_sequence_name] = "".join(sequence_lines)
    
    return sequences

def setup_model() -> tuple:

    model = AutoModelForMaskedLM.from_pretrained("facebook/esm1v_t33_650M_UR90S_1")
    tokeniser = AutoTokenizer.from_pretrained("facebook/esm1v_t33_650M_UR90S_1")
    model.eval()
    vocab_size = model.get_input_embeddings().weight.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    
    return model, tokeniser, device

def get_mutation_effect_score(merged_df: pd.DataFrame, model, tokeniser, device, batch_size: int) -> pd.DataFrame:
    
    """
    Splits mutations into batches and runs each batch through ESM1v
    
    Inputs:
    - Merged DF (pd.DataFrame) - The dataframe containing all mutations together
    - Model - The ESM1v model
    - Tokeniser - The appropriate tokeniser for the ESM1v model
    - Device - CPU, GPU or MPS selected for your system
    - Batch size (int) - The number of mutations that will be processed in parallel
    
    Outputs:
    - Merged DF (pd.DataFrame) - Updated dataframe with mutation scores added
    """
    
    results = {}
    
    for sequence_name, group in merged_df.groupby("sequence_name"):
        
        wt_sequence = group.iloc[0]["wt_sequence"]
        mutant_sequences = []
        mutant_info = [] 
    
        for index, row in group.iterrows():
            
            mutation = row["mutation"]
            wt_resdue = mutation[0]
            position = int(mutation[1:-1])
            mutant_residue = mutation[-1]
            mutant_sequence = wt_sequence[:position - 1] + mutant_residue + wt_sequence[position:]
            mutant_sequences.append(mutant_sequence)
            mutant_info.append((index, position, mutation))
    
        for batch_index in range(0, len(mutant_sequences), batch_size):

            batch_sequences = [wt_sequence] + mutant_sequences[batch_index:batch_index + batch_size]
            batch_info = mutant_info[batch_index:batch_index + batch_size]

            inputs = tokeniser(batch_sequences, return_tensors = "pt", padding = True, truncation = True, max_length = 1024)
            inputs = {key: value.to(device) for key, value in inputs.items()}
    
            with torch.no_grad():
                
                outputs = model(**inputs)
    
            logits = outputs["logits"]
            log_probs = torch.nn.functional.log_softmax(logits, dim = -1)
        
            for mutation_index, (global_index, position, mutation) in enumerate(batch_info, start = 1):
                
                wt_token_id = inputs["input_ids"][0, position]
                mut_token_id = inputs["input_ids"][mutation_index, position]
                score_wt = log_probs[0, position, wt_token_id].item()
                score_mut = log_probs[mutation_index, position, mut_token_id].item()
                delta = score_mut - score_wt
                results[global_index] = delta
            
            print("Found scores for sequences ", batch_index + batch_size, " out of ", len(mutant_sequences))
        
    merged_df["mutation_effect_score"] = merged_df.index.map(lambda index: results.get(index, None))

    return merged_df

__main__()