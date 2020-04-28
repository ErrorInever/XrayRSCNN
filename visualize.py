import matplotlib.pyplot as plt
import numpy as np


def show_batch(batch):
    images = batch[0]
    labels = batch[1]
    #std = [0.9902, 1.0123, 1.0078]
    #mean = [-0.0136,  0.1156,  0.3373]
    fig, ax = plt.subplots(1, 5, figsize=(20, 10))

    for i, img in enumerate(images):
        img = img.permute(1, 2, 0).numpy()
        label = labels[i].numpy().squeeze(0)
        label = 'NORMAL' if label == 0 else 'PNEUMONIA'
        #img = std * img + mean
        ax[i].imshow((img * 255).astype(np.uint8))
        ax[i].axis('off')
        ax[i].set_title(label, fontsize=15, weight='bold')
    plt.show()
