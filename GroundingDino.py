import torch
import cv2
import numpy as np
import os
from groundingdino.util.inference import load_model, load_image, predict, annotate

class GroundingDINOImage:
    def __init__(self, model_config_path, model_checkpoint_path, device="cuda"):
        self.device = device
        self.model = load_model(model_config_path, model_checkpoint_path, device=device)

    def detect_objects(self, image_path, labels, box_threshold=0.35, text_threshold=0.25):
        # Load and preprocess the image
        image_source, image = load_image(image_path)

        # Perform detection
        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=" . ".join(labels),  # Combine labels into a single caption
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )

        # Annotate the image with detected boxes and labels
        annotated_image = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

        return annotated_image, boxes, logits, phrases

# Example usage
if __name__ == "__main__":
    # Paths to model config and checkpoint
    MODEL_CONFIG_PATH = "GroundingDINO/GroundingDINO_SwinT_OGC.py"
    MODEL_CHECKPOINT_PATH = "GroundingDINO/groundingdino_swint_ogc.pth"

    # Initialize the GroundingDINOImage model
    grounding_dino = GroundingDINOImage(
        model_config_path=MODEL_CONFIG_PATH,
        model_checkpoint_path=MODEL_CHECKPOINT_PATH,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Input image and labels
    IMAGE_PATH = "GroundingDINO/.asset/cat_dog.jpeg"  # testing
    LABELS = ["cat", "dog", "table"]  # testing

    # Detect objects in the image
    annotated_image, boxes, logits, phrases = grounding_dino.detect_objects(IMAGE_PATH, LABELS)

    # Save and display the annotated image
    # output_path = "GroundingDINO/.asset/annotated_image.jpg"
    # cv2.imwrite(output_path, annotated_image)
    # print(f"Annotated image saved to {output_path}")

    # Print detection results
    print("Detection Results:")
    for box, logit, phrase in zip(boxes, logits, phrases):
        print(f"Label: {phrase}, Confidence: {logit:.2f}, Box: {box}")