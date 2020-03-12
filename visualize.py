import matplotlib.pyplot as plt
import numpy as np


def show_batch(batch):
    images = batch[0]
    labels = batch[1]
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]
    fig, ax = plt.subplots(1, 5, figsize=(30, 10))

    for i, img in enumerate(images):
        img = img.permute(1, 2, 0).numpy()
        label = labels[i].numpy().squeeze(0)
        label = 'NORMAL' if label == 0 else 'PNEUMONIA'
        img = std * img + mean
        ax[i].imshow((img * 255).astype(np.uint8))
        ax[i].axis('off')
        ax[i].set_title(label, fontsize=15, weight='bold')
    plt.show()
