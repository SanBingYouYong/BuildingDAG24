import yaml

# Read the YAML file
with open('./models/single_encdec_test_results.yml', 'r') as file:
    results = yaml.safe_load(file)

# Parse the contents
predictions = results['predictions']
ground_truth = results['ground_truth']

# Now you have access to predictions and ground_truth as Python lists
for i in range(5):
    print(f"Prediction: {predictions[i]}, Ground Truth: {ground_truth[i]}")
