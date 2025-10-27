'''
Configuration file to train KGE models on the AKG triples
'''
import os
import random
import string

slurm = "SLURM_JOB_ID" in os.environ # check if running on slurm
if slurm:
    JOB_ID = os.environ["SLURM_JOB_ID"] # get job id
    print("Running on slurm with job id:", JOB_ID)
else:
    # Generate random job id
    JOB_ID = "".join(random.choices(
        string.ascii_letters + string.digits, k=8)) 

output_folder = "results/"
dataset_folder = '../AKG_dataset'
models_folder = "KGE_models/"

results_csv_file = os.path.join(output_folder, "results.csv")
            

common_parameters = {
    "job_id": JOB_ID,
    'output_folder': output_folder,
    'results_csv_file': results_csv_file,
    'models_folder': models_folder,
}

conf = {
    # Experiment 001: Train a KGE model on pruned AKG triples
    "001": {
        'model_name': "RotatE",
        'triples_path': os.path.join(dataset_folder, 'AKG_pruned.tsv'),
        'triples_test_path': os.path.join(dataset_folder, 'AKG_test.tsv'),
        'triple_verification': False,
        'batch_size': 4096,
        'embedding_dim': 256,
        'learning_rate': 0.001,
        'num_epochs': 1000,
        'relations_to_filter': []
    },
    # Experiment 002: Train a KGE model on raw AKG triples
    "002": {
        'model_name': "RotatE",
        'triples_path': os.path.join(dataset_folder, 'AKG_raw.tsv'),
        'triples_test_path': os.path.join(dataset_folder, 'AKG_test.tsv'),
        'triple_verification': False,
        'batch_size': 4096,
        'embedding_dim': 256,
        'learning_rate': 0.001,
        'num_epochs': 1000,
        'relations_to_filter': []
    },
}

