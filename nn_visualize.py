import yaml
import matplotlib.pyplot as plt

loss_log_path = "./models/model_DAGDataset100_100_5_20240302162040_loss.yml"

with open(loss_log_path, 'r') as file:
    loss_log = yaml.safe_load(file)

train_losses = loss_log['train_losses']
val_losses = loss_log['val_losses']

epochs = range(1, len(train_losses) + 1)

plt.plot(epochs, train_losses, 'b', label='Training loss')
plt.plot(epochs, val_losses, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('loss_plot.png')
