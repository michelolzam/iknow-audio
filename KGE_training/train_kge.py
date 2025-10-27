from pprint import pprint
import argparse
import os
import json

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator
import pandas as pd

from config import conf, common_parameters


def preprocess_dataframe(df, conf):
    """
    Preprocess the AKG dataframe by applying filters and normalization.
    """

    print("Number of rows in the dataset:", len(df))
    print(df.head())

    # Select only verified triples (response == "Yes")
    if conf['triple_verification']:
        df = df[df['response'].str.contains("Yes", na=False)]

    # Filter relations
    df = df[~df['relation'].isin(conf['relations_to_filter'])]
    print("Number of rows after filtering relations:", len(df))

    # Filter rows where tail is "sound" or "Sound"
    df = df[(df['tail'] != 'sound') & (df['tail'] != 'Sound')]

    # --- Post-processing of the triples ---

    # Replace any underscores in the entity and relation names with spaces
    df['head'] = df['head'].str.replace('_', ' ')
    df['tail'] = df['tail'].str.replace('_', ' ')
    df['relation'] = df['relation'].str.replace('_', ' ')
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['head', 'relation', 'tail'])
    
    # Reset the index
    df = df.reset_index(drop=True)
    print("Number of rows after post-processing:", len(df))

    # Convert all columns to lowercase
    df['head'] = df['head'].str.lower()
    df['tail'] = df['tail'].str.lower()
    df['relation'] = df['relation'].str.lower()
    
    # Remove any leading or trailing whitespace
    df['head'] = df['head'].str.strip()
    df['tail'] = df['tail'].str.strip()
    df['relation'] = df['relation'].str.strip()
    
    return df


def merge_dicts(dict1, dict2):
    """
    Merge two dictionaries in extending parameters for common keys.
    """
    merged = dict1.copy()  # Make a copy of the first dictionary to preserve it
    for key, value in dict2.items():
        if key in merged:
            if isinstance(value, dict) and isinstance(merged[key], dict):
                # If both values are dictionaries, recursively merge them
                merged[key] = merge_dicts(merged[key], value)
            else:
                # If not, overwrite the value
                merged[key] = value
        else:
            merged[key] = value
    return merged

def parse_args():
    parser = argparse.ArgumentParser()
    parser.description = "Training script for learning KGE models."
    parser.add_argument("--conf_id", required=True,
                        help="Configuration tag located\
                            in config.py for the experiment")
    
    args = parser.parse_args()
    return args

def main(conf):

    triples_file = conf['triples_path']
    
    # Name of folder to store the model to be trained
    model_output_folder = os.path.join(
        conf['models_folder'],
        conf['conf_id'],
        f'{conf["job_id"]}_{conf["model_name"]}_{conf["triple_verification"]}')
    
    os.makedirs(model_output_folder, exist_ok=True)
                                       
    # --- Load and Preprocess the Audio-centric Knowledge Graph data ---
    df = pd.read_csv(triples_file, sep='\t')
    
    # Preprocess the dataframe
    df = preprocess_dataframe(df, conf)

    # Save AKG training triples to a temporary file compatible with PyKEEN
    train_triples_file = os.path.join(model_output_folder, 'AKG_train_triples.tsv')

    pd.DataFrame(
        {
        'subject':df['head'],
        'predicate':df['relation'],
        'object':df['tail'] 
        }
    ).to_csv(triples_file, sep='\t', header=False, index=False) 

    # --- Create TriplesFactory ---
    triples_factory = TriplesFactory.from_path(train_triples_file)

    # Create training, validation, and testing sets
    training_set = triples_factory

    test_triples = pd.read_csv(conf['triples_test_path'], sep='\t')

    # Save AKG test triples to a temporary file compatible with PyKEEN
    test_triples_file = os.path.join(model_output_folder, 'AKG_test_triples.tsv')

    pd.DataFrame(
        {
        'subject':test_triples['head'],
        'predicate':test_triples['relation'],
        'object':test_triples['tail'] 
        }
    ).to_csv(test_triples_file, sep='\t', header=False, index=False)

    # Create a TriplesFactory for the test set
    testing_set = TriplesFactory.from_path(
        test_triples_file,
        entity_to_id=triples_factory.entity_to_id,
        relation_to_id=triples_factory.relation_to_id
    )

    # Display the first few triples in the AKG training set
    triples_factory.triples

    # --- Run the PyKEEN pipeline ---
    result = pipeline(
        model=conf['model_name'],
        training=training_set,
        testing=testing_set,
        model_kwargs={
            'embedding_dim': conf['embedding_dim'],  
        },
        optimizer_kwargs={
            'lr': conf['learning_rate'],
        },
        training_kwargs={
            'num_epochs': conf['num_epochs'],
            'batch_size': conf['batch_size'],
        },
        random_seed=42,
        device='cuda'  # or 'cpu'
    )

    # Save the trained model
    result.save_to_directory(model_output_folder)
    print(f"Model saved to {model_output_folder}")

    # Create an evaluator and compute metrics
    evaluator = RankBasedEvaluator()

    metrics = evaluator.evaluate(
        result.model,
        testing_set.mapped_triples,
        additional_filter_triples=[training_set.mapped_triples]
    )

    # Print metrics
    print(f"Hits@1: {metrics.get_metric('hits@1')}")
    print(f"Hits@3: {metrics.get_metric('hits@3')}")
    print(f"Hits@5: {metrics.get_metric('hits@5')}")
    print(f"Hits@10: {metrics.get_metric('hits@10')}")
    print(f"Mean Reciprocal Rank: {metrics.get_metric('mean_reciprocal_rank')}")

    # Save metrics to an existing csv file saving training results
    results_csv_file = conf['results_csv_file']

    # Checking existence of the file to store results
    try:
        df = pd.read_csv(results_csv_file)
    except FileNotFoundError:
        # If the file does not exist, create a new DataFrame
        df = pd.DataFrame(columns=['job_id',
                                   'conf_id',
                                   'model_name',
                                   'triples_version',
                                   'verified',
                                   'hits@1',
                                   'hits@3',
                                   'hits@5',
                                   'hits@10',
                                   'mean_reciprocal_rank'])

    # Append the metrics to the DataFrame
    new_row = {
        'job_id': conf['job_id'],
        'conf_id': conf['conf_id'],
        'model_name': conf['model_name'],
        'verified': conf['triple_verification'],
        'hits@1': metrics.get_metric('hits@1'),
        'hits@3': metrics.get_metric('hits@3'),
        'hits@5': metrics.get_metric('hits@5'),
        'hits@10': metrics.get_metric('hits@10'),
        'mean_reciprocal_rank': metrics.get_metric('mean_reciprocal_rank')
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    # Save the DataFrame to a CSV file
    df.to_csv(results_csv_file, index=False)
    print(f"Results saved to {results_csv_file}")


if __name__ == '__main__':

    # Parse arguments
    args = parse_args()
    args = vars(args)
    print('Input arguments: ', args)

    conf = merge_dicts(common_parameters, conf[args["conf_id"]])
    conf = {**conf, **args}
    
    pprint(conf)

    # Save the configuration to a JSON file
    config_file_path = os.path.join(conf['models_folder'],
                                conf['conf_id'],
                                conf['job_id'])
    # Make folder if it does not exist
    os.makedirs(config_file_path, exist_ok=True)

    # Save as config.json
    with open(os.path.join(config_file_path, 'config.json'), 'w') as f:
        json.dump(conf, f, indent=4)
        print(f"Configuration saved to {os.path.join(config_file_path, 'config.json')}")


    main(conf)
