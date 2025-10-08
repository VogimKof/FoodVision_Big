from pathlib import Path
from torchvision import datasets, transforms
import torchvision
import os
import requests
import time
import shutil

def get_food101(    
    transform: transforms.Compose,
    data_path: str | Path,
) -> tuple[datasets.Food101, datasets.Food101, list[str]]:
    """
    Loads Food101 training and testing data, downloading if not already present.

    Args:
        data_path: Path or string to the dataset directory.
        transforms: A torchvision transforms.Compose object.

    Returns:
        (train_data, test_data, class_names)
    """
    # Ensure Path object
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    # Training transforms with augmentation
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.TrivialAugmentWide(),
        transform,
    ])

    print(f"Preparing food101 dataset in {data_path}...")

    # Load datasets
    train_data = datasets.Food101(
        root=data_path,
        split="train",
        transform=train_transforms,
        download=True
    )

    test_data = datasets.Food101(
        root=data_path,
        split="test",
        transform=transform,
        download=True
    )

    # Ensure classes exist (works across torchvision versions)
    try:
        class_names = train_data.classes
    except AttributeError:
        class_names = datasets.Food101.classes  # fallback to static attribute

    return train_data, test_data, class_names

def load_classes_from_file(filepath : str | Path) -> list[str]:
    """
    Load class names from a text file (one class per line).
    
    Args:
        filepath: Path to the text file containing class names
    
    Returns:
        list: List of class names (empty lines and comments are ignored)
    """
    classes = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Ignore empty lines and comments
                if line and not line.startswith('#'):
                    classes.append(line)
        print(f"Loaded {len(classes)} classes from {filepath}")
        return classes
    except FileNotFoundError:
        print(f"ERROR: File '{filepath}' not found!")
        return []
    except Exception as e:
        print(f"ERROR reading file: {e}")
        return []


def get_images_from_pexels(output_dir: str | Path,
                images_per_class : int = 10,
                api_key: str | None = None,
                min_width: int = 400, 
                classes_file: str | Path = None,
                overwrite: bool = True
                ) -> dict:
    """
    Download non-food images from Pexels API with specified images per class.
    Get free API key from: https://www.pexels.com/api/
    
    Args:
        images_per_class: Number of images to download per class.
        output_dir: Directory to save images.
        api_key: Pexels API key (optional, can also be set via PEXELS_API_KEY env variable).
        min_width: Minimum image width.
        classes_file: Path to text file with class names (optional).
        overwrite: Whether to overwrite existing output directory.
    
    Returns:
        dict: Download statistics
    """

    output_path = Path(output_dir)
    
    # Check if output directory exists
    if output_path.exists():
        if overwrite:
            print(f"Removing existing directory: {output_dir}")
            shutil.rmtree(output_path)
        else:
            print(f"Directory '{output_dir}' exists, will skip existing images")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    API_KEY = api_key or os.getenv("PEXELS_API_KEY", None)
    
    if not API_KEY:
        print("ERROR: Please set your Pexels API key!")
        print("Get one free at: https://www.pexels.com/api/")
        return {}
    
    # Load queries from file or use defaults
    if classes_file:
        queries = load_classes_from_file(classes_file)
        if not queries:
            print("No classes loaded, using default queries")
            queries = [
                "nature", "city", "technology", "animals", "architecture",
                "landscape", "ocean", "mountains", "abstract", "cars", "devices"
            ]
    else:
        queries = [
            "nature", "city", "technology", "animals", "architecture",
            "landscape", "ocean", "mountains", "abstract", "cars", "decives"
        ]
    
    headers = {"Authorization": API_KEY}
    total_downloaded = 0
    stats = {
        "images_per_class": {},
        "total_downloaded": 0,
        "total_target": len(queries) * images_per_class,
        "failed_downloads": 0,
        "classes": queries
    }
    
    print(f"Starting download: {images_per_class} images per class")
    print(f"Total classes: {len(queries)}")
    print(f"Minimum width: {min_width}px\n")
    
    for query in queries:
        query_count = 0
        page = 1
        max_pages = 15
        
        print(f"Downloading '{query}' images...")
        
        while query_count < images_per_class and page <= max_pages:
            url = f"https://api.pexels.com/v1/search"
            params = {"query": query, "page": page, "per_page": 80}
            
            try:
                response = requests.get(url, headers=headers, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if not data.get("photos"): #break if there are no photos for this query
                    break
                
                for photo in data["photos"]:
                    if query_count >= images_per_class:
                        break
                    
                    # Get large image
                    img_url = photo["src"]["large2x"]
                    if photo["width"] < min_width:
                        continue
                    
                    # Create class subdirectory
                    class_dir = os.path.join(output_dir, query)
                    Path(class_dir).mkdir(parents=True, exist_ok=True)
                    
                    filename = os.path.join(class_dir, f"{query}_{photo['id']}.jpg")
                    
                    if os.path.exists(filename):
                        query_count += 1
                        continue
                    
                    try:
                        img_data = requests.get(img_url, timeout=15)
                        img_data.raise_for_status()
                        
                        with open(filename, "wb") as f:
                            f.write(img_data.content)
                        
                        query_count += 1
                        total_downloaded += 1
                        
                        time.sleep(0.05)
                        
                    except Exception as e:
                        stats["failed_downloads"] += 1
                        continue
                
                page += 1
                
            except Exception as e:
                print(f"  API error for '{query}': {e}")
                break
        
        stats["images_per_class"][query] = query_count
        print(f"Downloaded {query_count}/{images_per_class} images for '{query}'")
    
    stats["total_downloaded"] = total_downloaded
    
    print(f"\n{'='*50}")
    print(f"Download Complete!")
    print(f"Total downloaded: {total_downloaded}/{stats['total_target']}")
    print(f"{'='*50}")
    
    return stats

