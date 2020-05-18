import matplotlib.pyplot as plt
import numpy as np


def show_batch(batch):
    """
    shows images from batch
    :param batch: ``Tensor`` images
    """
    images = batch[0]
    labels = batch[1]
    fig, ax = plt.subplots(1, 5, figsize=(20, 10))

    for i, img in enumerate(images):
        img = img.permute(1, 2, 0).numpy()
        label = labels[i].numpy().squeeze(0)
        label = 'NORMAL' if label == 0 else 'PNEUMONIA'
        ax[i].imshow((img * 255).astype(np.uint8))
        ax[i].axis('off')
        ax[i].set_title(label, fontsize=15, weight='bold')
    plt.show()


def show_result(img, pred_class, prob):
    """
    Shows predicted scores on current image
    :param img: ``Tensor`` image
    :param pred_class: ``str`` predicted class
    :param prob: ``float`` predicted probability
    """
    img = img.detach().cpu().squeeze(0).permute(1, 2, 0)
    plt.axis('off')
    plt.imshow(img)
    plt.title('Predicted class: {}\nProbability: {}%'.format(pred_class, round(prob, 1)))
    plt.savefig('result.png', bbox_inches='tight')
    plt.show()
