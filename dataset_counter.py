import os


path = "./datasets/DAGDataset5_5_5/images"

# removes any file in the folder if it is not a .png file
for file in os.listdir(path):
    if file == "pred":
        continue
    if not file.endswith(".png"):
        os.remove(os.path.join(path, file))


