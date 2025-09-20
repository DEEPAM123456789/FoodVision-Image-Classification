
### 1. Imports and class names setup ###
import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
with open("class_names.txt", 'r') as f:
    class_names = [food_name.strip() for food_name in f.readlines()]

### 2. Model and transforms preparation ###
# Create model and transforms
effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=101)

# Load saved weights
effnetb2.load_state_dict(
    torch.load(f="09_pretrained_effnetb2_feature_extractor_food101_20_percent.pth",
              map_location=torch.device("cpu")) # load to cpu
)

### 3. predict function ###
def predict(img) -> Tuple[Dict, float]:
    # Start a timer
    start_time = timer()

    # Transform the input image for use with EffNetB2
    img = effnetb2_transforms(img).unsqueeze(0) # unsqueeze = add batch dimension on 0th index

    # Put the model into eval mode, make prediction
    effnetb2.eval()
    with torch.inference_mode():
        # Pass transformerd image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(effnetb2(img), dim=1)

    # Create a prediction label and pediction probability dictionary
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    # Calculate pred time
    end_time = timer()
    pred_time = round(end_time - start_time, 4) 

    # Return pred dict and pred time
    return pred_labels_and_probs, pred_time

### 4. Gradio app ###

import gradio as gr

# Create title, description and article
title = "FoodVision Big 🍔👁️💪"
description = "An [EfficientNetB2 feature extractor](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b2.html#torchvision.models.efficientnet_b2) computer vision model to classify images 101 classes of food from the Food101 dataset."
article = "Created at 09. [PyTorch Model Deloyment](https://www.learnpytorch.io/09_pytorch_model_deployment/#11-turning-our-foodvision-big-model-into-a-deployable-app)"

# Create example list
example_list = [["example/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(fn=predict, # maps inputs to outputs
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=5, label="predictions"),
                             gr.Number(label="Prediction time (s)")],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
demo.launch()        
