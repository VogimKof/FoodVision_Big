import gradio as gr
import os
import torch
from pathlib import Path
from func.model_setup import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict
from dotenv import load_dotenv

load_dotenv()

food101_clases_file = Path(os.getenv("CLASSES_DIR")) / "food_classes.txt"
models_weigts_file = Path(os.getenv("MODEL_DIR"))
examples_dir = Path(os.getenv("DATA_DIR")) / "examples" 

# Setup class names
with open(food101_clases_file, "r") as f: 
    class_names = [food_name.strip() for food_name in  f.readlines()]
    
### Model and transforms preparation ###    

#Create model to classify if image is a food or not
effnetb2_isfood, effnetb2_isfood_transforms = create_effnetb2_model(
    num_classes=2)

# Create model to classify food type
effnetb2_food_type, effnetb2_food_type_transforms = create_effnetb2_model(
    num_classes=101, # could also use len(class_names)
)

# Load saved weights
effnetb2_food_type.load_state_dict(
    torch.load(
        f=models_weigts_file / "09_pretrained_effnetb2_feature_extractor_food101_20_percent.pth",
        map_location=torch.device("cpu"), 
        weights_only=True 
    )
)

effnetb2_isfood.load_state_dict(
    torch.load(
        f=models_weigts_file / "09_pretrained_effnetb2_feature_extractor_food.pth",
        map_location=torch.device("cpu"),
        weights_only=True
    )
)

### Predict function ###

def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()
    
    # Step 1: Classify image as food or not food
    img_transformed_isfood = effnetb2_isfood_transforms(img).unsqueeze(0)
    effnetb2_isfood.eval()
    with torch.inference_mode():
        pred = torch.softmax(effnetb2_isfood(img_transformed_isfood), dim=1)
    
    # If not food, return early
    if torch.argmax(pred, dim=1) == 1: # 0 = not food, 1 = food
        pred_time = round(timer() - start_time, 5)
        return {"food": float(pred[0][0]), "not food": float(pred[0][1])}, pred_time
    
    # Step 2: If food, classify food type using the ORIGINAL PIL image
    img_transformed_food = effnetb2_food_type_transforms(img).unsqueeze(0)
    
    # Put model into evaluation mode and turn on inference mode
    effnetb2_food_type.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model
        pred_probs = torch.softmax(effnetb2_food_type(img_transformed_food), dim=1)
    
    # Create prediction dictionary
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)
    
    # Return the prediction dictionary and prediction time 
    return pred_labels_and_probs, pred_time

### Gradio app ###

# Create title, description and article strings
title = "FoodVision Big 🍔👁"
description = "An EfficientNetB2 feature extractor computer vision model to classify images of food into [101 different classes](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/food101_class_names.txt)."

# Create examples list from "examples/" directory
example_list = [[str(examples_dir / example)] for example in os.listdir(examples_dir)]

# Create Gradio interface 
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=5, label="Predictions"),
        gr.Number(label="Prediction time (s)"),
    ],
    examples=example_list,
    title=title,
    description=description,
)

demo.launch()