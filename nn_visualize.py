import yaml
import matplotlib.pyplot as plt


def visualize_loss(loss_path: str, curve_path: str=None):
    with open(loss_path, 'r') as file:
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

    # Set y-axis limits to +1 of second epoch loss
    ylim_value = val_losses[1] + 1 if len(val_losses) > 1 else val_losses[0]
    plt.ylim(0, ylim_value)

    pdf_path = curve_path if curve_path else loss_path.replace('.yml', '.pdf')

    plt.savefig(pdf_path)


if __name__ == "__main__":
    loss_path = "./models/model_DAGDataset100_100_5_20240323160407_loss.yml"
    visualize_loss(loss_path)
