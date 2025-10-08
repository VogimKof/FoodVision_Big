import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from datetime import datetime
import os
from PIL import Image
from torchvision import transforms

def save_model(model: torch.nn.Module,
               target_dir: str | Path,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
        model: A target PyTorch model to save.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.

    Example usage:
        save_model(model=model_0,
                target_dir="models",
                model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                            exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
                f=model_save_path)


def set_seeds(seed: int = 42) -> None:
    """Sets various random seeds for reproductibility.

    Args:
        seed: An integer to set all the random seeds to.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def create_writer(experiment_name: str, 
                  model_name: str, 
                  extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter:
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name: Name of experiment.
        model_name: Name of model.
        extra: Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)

def predict_and_display(model: torch.nn.Module, 
                        folder_path: str | Path, 
                        transform: transforms.Compose, 
                        class_names: list = None, 
                        device = "cpu", 
                        max_images: int = 12):
    """
    Run predictions on images in a folder, display them with predicted class and probability.
    
    Args:
        model: Trained PyTorch model.
        folder_path: Path to folder containing images.
        transform: torchvision transform for preprocessing.
        class_names: List of class names.
        device: 'cuda' or 'cpu'. Auto-detects if None.
        max_images: Max number of images to display.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()
    model.to(device)

    images, preds, probs = [], [], []

    with torch.no_grad():
        for filename in sorted(os.listdir(folder_path)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, filename)
                image = Image.open(img_path).convert('RGB')
                input_tensor = transform(image).unsqueeze(0).to(device)

                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                pred_idx = torch.argmax(probabilities, dim=1).item()
                prob = probabilities[0, pred_idx].item()

                pred_class = class_names[pred_idx] if class_names else str(pred_idx)

                images.append((filename, image))
                preds.append(pred_class)
                probs.append(prob)

                if len(images) >= max_images:
                    break

    #Display results 
    num_images = len(images)
    cols = 4
    rows = (num_images + cols - 1) // cols
    plt.figure(figsize=(4 * cols, 4 * rows))

    for i, (filename, img) in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{preds[i]} ({probs[i]*100:.1f}%)", fontsize=10)

    plt.tight_layout()
    plt.show()

