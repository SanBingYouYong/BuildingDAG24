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

    for category in DISCRETE_VARIABLES:
        if category not in results:
            continue
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
        print(f" - Accuracy: {100 * correct / total:.2f}%")



if __name__ == "__main__":
    results_path = "results.yml"
    with open(results_path, "r") as file:
        results = yaml.safe_load(file)["results"]


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
    for category in categories:
        print(f"Category: {category}")
        # # calculate accuracy
        # correct = 0
        # total = 0
        # for i, (pred, target) in enumerate(results[category]):
        #     pred = pred
        #     target = target
        #     if pred == target:
        #         correct += 1
        #     total += 1
        # print(f"Accuracy: {100 * correct / total:.2f}%")
        # F1
        f1 = f1_score([target for _, target in results[category]], [pred for pred, _ in results[category]], average="weighted")
        print(f"F1: {f1:.2f}")



