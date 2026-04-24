import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(losses_train, errors_train, mode, model_type):
    epochs = range(1, len(losses_train) + 1) # sp we start 1 and end at the right number
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5)) # make subplots baby

    # --- loss curve ---
    ax1.plot(epochs, losses_train, label='Train Loss', color='steelblue')
    # rolling average to show the trend through the noise
    window = 5
    rolling_avg = np.convolve(losses_train, np.ones(window)/window, mode='valid')
    ax1.plot(range(window, len(losses_train)+1), rolling_avg,
             label=f'{window}-epoch rolling avg', color='red', linewidth=2)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('BCE Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # --- error curve ---
    ax2.plot(epochs, errors_train, label='Train Error %', color='orange')
    rolling_err = np.convolve(errors_train, np.ones(window)/window, mode='valid')
    ax2.plot(range(window, len(errors_train)+1), rolling_err,
             label=f'{window}-epoch rolling avg', color='red', linewidth=2)
    ax2.set_title('Training Error Rate')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Error %')
    ax2.legend()
    ax2.grid(alpha=0.3)

    title = f"MLP training curves, mode {mode}, model {model_type}"
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    # plt.savefig('mlp_training_curves.png', dpi=150)
    plt.show()