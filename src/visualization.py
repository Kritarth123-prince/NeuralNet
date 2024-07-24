import matplotlib.pyplot as plt

def plot_training_history(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy / Loss')
    plt.legend()
    plt.show()

def visualize_images(images, labels=None):
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(20, 20))
    for i, img in enumerate(images):
        if labels is not None:
            axes[i].set_title(labels[i])
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    plt.show()