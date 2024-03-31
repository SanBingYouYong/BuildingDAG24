import yaml
import matplotlib.pyplot as plt

# Load data from YAML file
yml_path = "./performance.yml"
with open(yml_path, "r") as file:
    performances = yaml.safe_load(file)

# Extract data for each metric
f1_data = performances['f1']
mae_data = performances['mae']

# Sort metrics based on scores
sorted_f1_data = sorted(f1_data.items(), key=lambda x: x[1])
sorted_mae_data = sorted(mae_data.items(), key=lambda x: x[1], reverse=True)

# Extract sorted keys and values
f1_keys = [item[0] for item in sorted_f1_data]
f1_values = [item[1] for item in sorted_f1_data]

mae_keys = [item[0] for item in sorted_mae_data]
mae_values = [item[1] for item in sorted_mae_data]

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Subplot 1 for F1 metrics
axs[0].barh(f1_keys, f1_values, color='skyblue')
axs[0].set_xlabel('F1 Scores')
axs[0].set_title('F1 Metrics')

# Subplot 2 for MAE metrics
axs[1].barh(mae_keys, mae_values, color='salmon')
axs[1].set_xlabel('MAE Scores')
axs[1].set_title('MAE Metrics')

plt.tight_layout()
plt.savefig("performance.pdf")