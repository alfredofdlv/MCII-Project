import matplotlib.pyplot as plt

def plot_DL_results(
    train_accuracies,
    valid_accuracies,
    train_losses,
    valid_losses,
    train_f1_scores,
    valid_f1_scores,
    train_recalls,
    valid_recalls,
    epochs_range=None
):
    # Determine epochs_range if not provided
    if epochs_range is None:
        epochs_range = range(1, len(train_accuracies) + 1)
    
    # Plotting the loss
    plt.figure(figsize=(12, 6))
    
    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, valid_losses, label='Validation Loss', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.grid(True)

    # Metrics subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy', color='g')
    plt.plot(epochs_range, valid_accuracies, label='Validation Accuracy', color='y')
    plt.plot(epochs_range, train_f1_scores, label='Train F1 Score', color='b')
    plt.plot(epochs_range, valid_f1_scores, label='Validation F1 Score', color='orange')
    plt.plot(epochs_range, train_recalls, label='Train Recall', color='r')
    plt.plot(epochs_range, valid_recalls, label='Validation Recall', color='purple')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('Metrics per Epoch')
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
