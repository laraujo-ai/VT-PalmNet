"""
Generic training utilities for PalmNet.
(Domain-specific CCNet helpers — Gabor visualisation, hook registration — are not included.)
"""

import os
import pickle

import matplotlib.pyplot as plt
plt.switch_backend('agg')


def get_file_names(txt: str) -> list[str]:
    """Parse a dataset .txt file and return the list of image paths."""
    paths = []
    with open(txt, 'r') as f:
        for line in f:
            paths.append(line.strip().split(' ')[0])
    return paths


def save_loss_acc(
    train_losses: list,
    train_accuracy: list,
    best_acc: float,
    path_rst: str,
):
    """Save loss / accuracy logs as .txt and .pickle, and plot curves."""
    os.makedirs(path_rst, exist_ok=True)

    # --- pickle ---
    for name, obj in [
        ('train_losses',   train_losses),
        ('train_accuracy', train_accuracy),
    ]:
        with open(os.path.join(path_rst, f'{name}.pickle'), 'wb') as f:
            pickle.dump(obj, f)

    # --- txt ---
    with open(os.path.join(path_rst, 'train_losses.txt'), 'w') as f:
        f.writelines(f'{v}\n' for v in train_losses)
    with open(os.path.join(path_rst, 'train_accuracy.txt'), 'w') as f:
        f.writelines(f'{v}\n' for v in train_accuracy)
    with open(os.path.join(path_rst, 'best_train_accuracy.txt'), 'w') as f:
        f.write(str(best_acc))

    # --- plots ---
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, 'b', label='train loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.savefig(os.path.join(path_rst, 'losses.png'))
    plt.close()

    plt.figure()
    plt.plot(epochs, train_accuracy, 'b', label='train accuracy')
    plt.axhline(y=best_acc, color='r', linestyle='--', label=f'best {best_acc:.2f}%')
    plt.legend()
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('accuracy (%)')
    plt.savefig(os.path.join(path_rst, 'accuracy.png'))
    plt.close()
