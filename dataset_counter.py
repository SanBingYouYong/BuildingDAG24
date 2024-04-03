import os


path = "./datasets/DAGDatasetDistorted100_100_5"
images_path = os.path.join(path, "images")
params_path = os.path.join(path, "params")

# # removes any file in the folder if it is not a .png file
# for file in os.listdir(path):
#     if file == "pred":
#         continue
#     if not file.endswith(".png"):
#         os.remove(os.path.join(path, file))

# count images
images_count = 0
params_count = 0
for file in os.listdir(images_path):
    if file == "pred":
        continue
    if file.endswith(".png"):
        images_count += 1
for file in os.listdir(params_path):
    if file.endswith(".yml"):
        params_count += 1
print(f"Number of images: {images_count}")
print(f"Number of params: {params_count}")
