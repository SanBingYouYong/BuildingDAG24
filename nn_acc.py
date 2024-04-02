import yaml
from sklearn.metrics import f1_score


DISCRETE_VARIABLES = [
    "Bm Base Shape",
    "Rf Base Shape",
    "Num Floors",
    "Num Windows Each Side",
    "Has Floor Ledge",
    "Has Window Ledge",
    "Window Divided Horizontal",
    "Window Divided Vertical",
]

def acc_discrete(results_path: str="results.yml"):
    print(f"Calculating accuracies for discrete variables in {results_path}")
    with open(results_path, "r") as file:
        results = yaml.safe_load(file)["results"]

    # for category in DISCRETE_VARIABLES:
    #     if category not in results:
    #         continue
    #     print(f"Category: {category}")
    #     # calculate accuracy
    #     correct = 0
    #     total = 0
    #     for i, (pred, target) in enumerate(results[category]):
    #         pred = pred
    #         target = target
    #         if pred == target:
    #             correct += 1
    #         total += 1
    #     print(f" - Accuracy: {100 * correct / total:.2f}%")
    f1scores = {}
    maes = {}
    for cat in results.keys():
        if cat in DISCRETE_VARIABLES: # discrete var
            print(f"Category: {cat}")
            # F1
            f1 = f1_score([target for _, target in results[cat]], [pred for pred, _ in results[cat]], average="weighted")
            print(f"F1: {f1:.2f}")
            f1scores[cat] = float(f1)
        else:
            # for continuous var, we calculate MAE
            print(f"Category: {cat}")
            # mae = sum([abs(target - pred) for pred, target in results[cat]]) / len(results[cat])
            # check for vector outputs
            if isinstance(results[cat][0][0], list):
                mae = sum([sum([abs(target - pred) for pred, target in zip(preds, targets)]) / len(preds) for preds, targets in results[cat]]) / len(results[cat])
            else:
                mae = sum([abs(target - pred) for pred, target in results[cat]]) / len(results[cat])
            print(f"MAE: {mae:.2f}")
            maes[cat] = float(mae)
    
    with open("performance.yml", "w") as file:
        yaml.dump({"f1": f1scores, "mae": maes}, file)



if __name__ == "__main__":
    results_path = "results.yml"
    with open(results_path, "r") as file:
        results = yaml.safe_load(file)["results"]

    f1scores = {}
    maes = {}

    categories = [
        "Bm Base Shape",
        "Rf Base Shape",
        "Num Floors", 
        "Num Windows Each Side", 
        "Has Floor Ledge",
        "Has Window Ledge",
        "Window Divided Horizontal",
        "Window Divided Vertical"
    ]
    # for category in categories:
    #     print(f"Category: {category}")
    #     # F1
    #     f1 = f1_score([target for _, target in results[category]], [pred for pred, _ in results[category]], average="weighted")
    #     print(f"F1: {f1:.2f}")
    
    for cat in results.keys():
        if cat in categories: # discrete var
            print(f"Category: {cat}")
            # F1
            f1 = f1_score([target for _, target in results[cat]], [pred for pred, _ in results[cat]], average="weighted")
            print(f"F1: {f1:.2f}")
            f1scores[cat] = float(f1)
        else:
            # for continuous var, we calculate MAE
            print(f"Category: {cat}")
            # mae = sum([abs(target - pred) for pred, target in results[cat]]) / len(results[cat])
            # check for vector outputs
            if isinstance(results[cat][0][0], list):
                mae = sum([sum([abs(target - pred) for pred, target in zip(preds, targets)]) / len(preds) for preds, targets in results[cat]]) / len(results[cat])
            else:
                mae = sum([abs(target - pred) for pred, target in results[cat]]) / len(results[cat])
            print(f"MAE: {mae:.2f}")
            maes[cat] = float(mae)
    
    with open("performance.yml", "w") as file:
        yaml.dump({"f1": f1scores, "mae": maes}, file)




