import torch
import torchvision
from torch import nn

def create_effnetb2_model(num_classes: int = 3, 
                          weights_path = None, 
                          seed: int = 42 ) -> tuple[nn.Module, torchvision.transforms.Compose]:
    """Creates an EfficientNetB2 feature extractor model and transforms.

    Args:
        num_classes: number of classes in the classifier head. 
        weights_path: path to model weights. Defaults to None.
        seed: random seed value.

    Returns:
        model (torch.nn.Module): EffNetB2 feature extractor model. 
        transforms (torchvision.transforms): EffNetB2 image transforms.
    """
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transforms = weights.transforms()

    # Start with pretrained ImageNet weights (better than random init)
    model = torchvision.models.efficientnet_b2(weights=weights)

    # Freeze backbone only
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier
    torch.manual_seed(seed)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=num_classes),
    )

    # Load custom weights if provided
    if weights_path is not None:
        state_dict = torch.load(weights_path, weights_only=True)
        model.load_state_dict(state_dict)

    return model, transforms
