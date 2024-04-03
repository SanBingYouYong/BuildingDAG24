import yaml
import os

def process_yaml_files(directory):
    architecture_metrics = {}

    # Iterate over each YAML file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".yml"):
            filepath = os.path.join(directory, filename)
            architecture_name = os.path.splitext(filename)[0]
            with open(filepath, 'r') as file:
                data = yaml.safe_load(file)
                f1_values = list(data['f1'].values())
                mae_values = list(data['mae'].values())

                # Calculate min, max, and average for F1
                f1_min = min(f1_values)
                f1_max = max(f1_values)
                f1_avg = sum(f1_values) / len(f1_values)

                # Calculate min, max, and average for MAE
                mae_min = min(mae_values)
                mae_max = max(mae_values)
                mae_avg = sum(mae_values) / len(mae_values)

                # Store the results
                architecture_metrics[architecture_name] = {
                    'f1': {'min': f1_min, 'max': f1_max, 'avg': f1_avg},
                    'mae': {'min': mae_min, 'max': mae_max, 'avg': mae_avg}
                }

    return architecture_metrics

def save_averaged_metrics(averaged_metrics, output_file):
    with open(output_file, 'w') as file:
        yaml.dump(averaged_metrics, file)

if __name__ == "__main__":
    directory = "performances"
    averaged_metrics = process_yaml_files(directory)
    save_averaged_metrics(averaged_metrics, "performance_averages.yml")
