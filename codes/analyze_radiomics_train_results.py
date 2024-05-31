import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def format_data(data, model_type, feature_aggregation):
    formatted_data = {
        "model_type": model_type,
        "feature_aggregation": feature_aggregation,
        "selector": data["selector"],
        "classifier": data["classifier"],
        "average_auc": round(data["average_auc"], 3),
        "std_dev_auc": round(data["std_dev_auc"], 3),
        "hyperparameters": data["hyperparameters"],
        "features": data["features"],
        "model_coefficients": data.get("model_coefficients", ""),
        "feature_importances": data.get("feature_importances", ""),
        "model_intercept": data.get("model_intercept", ""),
    }
    return formatted_data

def create_csv_from_json(main_directory):
    all_data = []
    for model_type in ['radiomics']:
        model_path = os.path.join(main_directory, model_type)
        
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if file.endswith('.json'):
                    feature_aggregation = os.path.basename(root) if model_type != 'clinical'and model_type != 'combined' else ''
                    file_path = os.path.join(root, file)
                    data = load_json(file_path)
                    formatted_data = format_data(data, model_type, feature_aggregation)
                    all_data.append(formatted_data)
    df = pd.DataFrame(all_data)
    output_csv_path = os.path.join(main_directory, 'train_results.csv')
    df.to_csv(output_csv_path, index=False)
    return output_csv_path

main_directory = "/nfs/rnas/clolivia/_Experiments/immuno_paper/laia/train_results"
csv_file_path = create_csv_from_json(main_directory)
print(f"CSV file created at: {csv_file_path}")