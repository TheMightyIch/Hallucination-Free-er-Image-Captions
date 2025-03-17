from Models.AbstractModel import AbstractModel
import os
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate

class groundingDINO(AbstractModel):
    def __init__(self,model_name: str):
        super().__init__(model_name)

    def generateModel(self, **inputs):
        return load_model(self.config["config_path"],os.getcwd()+"/"+ self.config["model_id"], device=self.DEVICE)

    def generateResponse(self, image_path, labels, box_threshold=0.35, text_threshold=0.25):
        # Load and preprocess the image
        image_source, image = load_image(image_path)

        # Perform detection
        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=" . ".join(labels),  # Combine labels into a single caption
            box_threshold=self.config["box_threshold"],
            text_threshold=self.config["text_threshold"],
            device=self.DEVICE
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
    grounding_dino = groundingDINO("groundingDINO")

    import cv2
    import numpy as np
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
