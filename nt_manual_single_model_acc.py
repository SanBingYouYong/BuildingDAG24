import yaml


results_path = "./models/single_encdec_test_results.yml"
with open(results_path, "r") as file:
    results = yaml.safe_load(file)["results"]

# calculate accuracy
correct = 0
total = 0
for i, (pred, target) in enumerate(results):
    pred = pred
    target = target
    if pred == target:
        correct += 1
    total += 1
print(f"Accuracy: {100 * correct / total:.2f}%")



