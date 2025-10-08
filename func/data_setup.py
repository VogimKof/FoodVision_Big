import os
import torch
import torchvision
from torch.utils.data import DataLoader
from pathlib import Path
import random
import shutil

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_data: torchvision.datasets, 
    test_data: torchvision.datasets,
    batch_size: int, 
    num_workers: int = NUM_WORKERS
) -> tuple[DataLoader, DataLoader]: 
    """Creates training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        transform: torchvision transforms to perform on training and testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.

    Returns:
        A tuple of (train_dataloader, test_dataloader).
        Example usage:
        train_dataloader, test_dataloader, class_names = \
            = create_dataloaders(train_dir=path/to/train_dir,
                                test_dir=path/to/test_dir,
                                transform=some_transform,
                                batch_size=32,
                                num_workers=4)
    """

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False, # don't need to shuffle test data
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader

def copy_images(image_paths, dest_dir):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for img_path in image_paths:
        shutil.copy(img_path, dest_dir)

def create_pytorch_dataset(food_dir: str | Path, 
                           nonfood_dir: str | Path, 
                           output_dir: str | Path, 
                           train_ratio: float = 0.8, 
                           seed: int= 42,
                           overwrite=False, 
                           max_food_images=None, #per class 
                           max_nonfood_images=None) -> dict : #per class
    """
    Create PyTorch-style dataset structure with train/test splits.
    
    Directory structure created:
    dataset/
    ├── train/
    │   ├── food/
    │   └── non_food/
    └── test/
        ├── food/
        └── non_food/
    
    Args:
        food_dir: Path to food images directory.
        nonfood_dir: Path to non-food images directory.
        output_dir: Output directory for dataset.
        train_ratio: Proportion for training set (default: 0.8, test will be 0.2).
        seed: Random seed for reproducibility.
        overwrite: If True, delete existing dataset directory and recreate.
        max_food_images: Maximum number of food images to use (None = use all).
        max_nonfood_images: Maximum number of non-food images to use (None = use all).

    Returns:
        dict: Statistics of images processed and split.
    """
    # Validate ratio
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("train_ratio must be between 0 and 1")
    
    output_path = Path(output_dir)
    
    # Check if output directory exists
    if output_path.exists():
        if overwrite:
            print(f"Removing existing directory: {output_dir}")
            shutil.rmtree(output_path)
        else:
            print(f"Directory '{output_dir}' already exists!")
            print("Options:")
            print("Use a different output_dir | Set overwrite=True to delete and recreate | Manually delete the directory")
            return None
    
    random.seed(seed)
    
    # Create output directory structure
    splits = ['train', 'test']
    classes = ['food', 'non_food']
    
    for split in splits:
        for cls in classes:
            Path(output_dir, split, cls).mkdir(parents=True, exist_ok=True)
    
    stats = {
        'train': {'food': 0, 'non_food': 0},
        'test': {'food': 0, 'non_food': 0},
    }
    
    #traverse food directory and each class subdirectory
    #foreach class subdirectory, shuffle images, limit to max_food_images if set
    #split based on train_ratio
    #copy images to output_dir/train/food or output_dir/test/food based on train_ratio

    for category, src_dir, max_images in [('food', food_dir, max_food_images), 
                                        ('non_food', nonfood_dir, max_nonfood_images)]:
        
        print(f"Processing {category} images...")
        src_dir = Path(src_dir)
        
        # If multiple classes inside the category
        class_dirs = [d for d in src_dir.iterdir() if d.is_dir()]
        
        for cls_dir in class_dirs:
            images = list(cls_dir.glob('*.jpg*'))  # all files, you could filter by .jpg/.png
            random.shuffle(images)
            total_available = len(images)
            
            if max_images is not None:
                
                max = min(max_images, total_available)

                if total_available < max_images:
                    print(f"Warning: Only {total_available} images available in {cls_dir}, less than max_images={max_images}. Using all available images.")
                
                images = images[:max] 
            
            n_train = int(len(images) * train_ratio)
            train_images = images[:n_train]
            test_images = images[n_train:]
            
            # Copy images
            copy_images(train_images, Path(output_dir, 'train', category))
            copy_images(test_images, Path(output_dir, 'test', category))
            
            # Update stats
            stats['train'][category] += len(train_images)
            stats['test'][category] += len(test_images)

            print(f"Class '{cls_dir.name}': {len(train_images)} train, {len(test_images)} test images.")

    return stats

def split_dataset(dataset: torchvision.datasets,
                  split_size: float = 0.2, 
                  seed: int = 42) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Randomly splits a given dataset into two proportions based on split_size and seed.

    Args:
        dataset: A PyTorch Dataset, typically one from torchvision.datasets.
        split_size: How much of the dataset should be split? 
            E.g. split_size=0.2 means there will be a 20% split and an 80% split. Defaults to 0.2.
        seed: Seed for random generator. Defaults to 42.

    Returns:
        tuple: (random_split_1, random_split_2) where random_split_1 is of size split_size*len(dataset) and 
            random_split_2 is of size (1-split_size)*len(dataset).
    """
    # Create split lengths based on original dataset length
    length_1 = int(len(dataset) * split_size) # desired length
    length_2 = len(dataset) - length_1 # remaining length
        
    # Print out info
    print(f"Splitting dataset of length {len(dataset)} into splits of size: {length_1} ({int(split_size*100)}%), {length_2} ({int((1-split_size)*100)}%)")
    
    # Create splits with given random seed
    random_split_1, random_split_2 = torch.utils.data.random_split(dataset, 
                                                                   lengths=[length_1, length_2],
                                                                   generator=torch.manual_seed(seed)) # set the random seed for reproducible splits
    return random_split_1, random_split_2