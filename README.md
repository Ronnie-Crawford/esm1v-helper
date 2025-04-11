# esm1v-helper
A script to help run esm1v at scale.

## User guide
I hope to make this as user-firendly as possible, if anything is unclear let me know.

### Steps for setting up
1) Download this Github repo
2) If using conda, set up the environment using ```conda env create -f environment.yml```, all packages should be available from unlicenced conda channels

### Fill in config
There are 4 variables for you to set before use:
1) ```fasta_file_path``` - This is the path to a fasta file containing your WT sequences, and their names
2) ```mutation_folder_path``` - This is the path to a folder containing ```.txt``` files, it expects 1 file for each domain, each with a list of mutations and the name of their associated WT
3) ```batch_size``` - This is the number of mutants that will be processed together, a larger number will make things run faster but will require more memory, can tailor it depending on the compute your have available
4) ```results_path``` - This is the path to the file where the results will be saved, if this file already exists, it will be overwritten

### Run
Then simply to run the script, use ```python path/to/esm1v_helper.py``` putting in the correct path to the esm1v helper script

