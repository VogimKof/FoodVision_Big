import matplotlib.pyplot as plt
import random
from PIL import Image
from pathlib import Path
from torchvision import transforms
import torch

def plot_transformed_images(image_dir: str | Path,
                            transform: transforms.Compose | None, 
                            n: int = 3, 
                            seed: int = 42):
    """Plots a series of random images from image_dir.

    Will open n image directory from image_dir, transform them
    with transform and plot them side by side.

    Args:
        image_paths: Directory with images inside. 
        transform: Transforms to apply to images.
        n: Number of images to plot.
        seed: Random seed for the random generator.
    """
    image_dir = Path(image_dir)
    
    # Get all image files from directory (including subdirectories)
    image_paths = list(image_dir.rglob('*.jpg')) + \
                  list(image_dir.rglob('*.jpeg')) + \
                  list(image_dir.rglob('*.png')) + \
                  list(image_dir.rglob('*.bmp'))
    
    # Sample random images
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(f) 
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            if transform:
                transformed_image = transform(f)
                
                # Denormalize if using ImageNet normalization
                if isinstance(transformed_image, torch.Tensor):
                    # Check if normalized (values outside [0,1])
                    if transformed_image.min() < 0 or transformed_image.max() > 1:
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        transformed_image = transformed_image * std + mean
                        transformed_image = torch.clamp(transformed_image, 0, 1)
                    
                    # Permute for matplotlib (C, H, W) -> (H, W, C)
                    transformed_image = transformed_image.permute(1, 2, 0).numpy()
                
                ax[1].imshow(transformed_image) 
                ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            else:
                ax[1].imshow(f)
                ax[1].set_title("No Transform")
            
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
            plt.tight_layout()
            plt.show()

def plot_loss_curves(results : dict):
    """Plots training curves of a results dictionary.

    Args:
        results: dictionary containing list of values, e.g.
            {"train_loss": [...],
            "train_acc": [...],
            "test_loss": [...],
            "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()