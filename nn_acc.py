import yaml


results_path = "results.yml"
with open(results_path, "r") as file:
    results = yaml.safe_load(file)["outputs"]

category = "Bm Base Shape"
print(f"Category: {category}")
# calculate accuracy
correct = 0
total = 0
for i, (pred, target) in enumerate(results[category]):
    pred = pred
    target = target
    if pred == target:
        correct += 1
    total += 1
print(f"Accuracy: {100 * correct / total:.2f}%")



