
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
from Models.AbstractModel import AbstractModel
import os
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate

class groundingDINO(AbstractModel):
    def __init__(self,model_name: str):
        super().__init__(model_name)

    def generateModel(self, **inputs):
        self.processor = AutoProcessor.from_pretrained(self.config["model_id"])
        model = AutoModelForZeroShotObjectDetection.from_pretrained(self.config["model_id"])
        return model

    def generateResponse(self, image,text):
        inputs = self.processor(images=image, text=text, is_split_into_words=self.config["tokenized"], return_tensors="pt").to(self.DEVICE)
        with torch.no_grad():
            outputs =self.model(**inputs)
        post_output = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.config["box_threshold"],
            text_threshold=self.config["text_threshold"],
            target_sizes=[image.size[::-1]]
        )[0]
        results=[]
        for box, labels in zip(post_output["boxes"], post_output["labels"]):
            results.append("Detected {} at location {}".format(labels, box.tolist()))
        return results

    def cleanModel(self):
        super().cleanModel()
        del self.processor

if __name__ == "__main__":
    import requests
    from RamPlusPlus import RamPlusPlus
    from PIL import Image
    ram = RamPlusPlus("RamPlusPlus")
    ram.loadModel()
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open("S:\phili\GitHub\Hallucination-Free-er-Image-Captions\image/AMBER_2.jpg")
    tags=ram.generateResponse(image)
    text_labels = [["a cat", "a remote control"]]
    print(text_labels)
    test = groundingDINO("groundingDINO")
    test.loadModel()
    # Check for cats and remote controls


    result = test.generateResponse(image,tags)
    for res in result:
        print(res)