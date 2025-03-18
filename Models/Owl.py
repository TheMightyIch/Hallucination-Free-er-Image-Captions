import numpy as np
import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from Models.AbstractModel import AbstractModel


class Owl(AbstractModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.autoProcessor =  OwlViTProcessor.from_pretrained(self.id)


    def generateModel(self, **inputs):
        return OwlViTForObjectDetection.from_pretrained(self.id)

    def generateResponse(self, labels : list [str], image):
        inputs= self.autoProcessor(text= labels, images=image, return_tensors="pt").to(self.DEVICE)
        with torch.no_grad():
            output = self.model(**inputs)
        unnormalized_image = self.get_preprocessed_image(inputs.pixel_values)
        target_sizes = torch.Tensor([unnormalized_image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to final bounding boxes and scores
        results = self.autoProcessor.post_process_object_detection(outputs=output, threshold=self.config["threshold"], target_sizes=target_sizes)
        return results

    def get_preprocessed_image(self,pixel_vals):
        pixel_values = pixel_vals.squeeze().cpu().numpy()
        unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:,None, None]
        unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
        unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
        unnormalized_image = Image.fromarray(unnormalized_image)
        return unnormalized_image

    def cleanModel(self):
        super().cleanModel()
        del self.autoProcessor


if __name__ == "__main__":
    import  RamPlusPlus
    owl = Owl("Owl")
    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    texts= [RamPlusPlus.test()]
    image=Image.open("../image/AMBER_2.jpg")
    text = texts[i]
    owl.loadModel()
    result= owl.generateResponse(texts, image)
    boxes, scores, labels = result[i]["boxes"], result[i]["scores"], result[i]["labels"]

    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
