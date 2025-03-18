from Models.AbstractModel import AbstractModel
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class test (AbstractModel):
    def __init__(self, model_name: str ):
        super().__init__(model_name)
        self.banana="apple"

    def generateModel(self, **inputs):
        self.processor = AutoProcessor.from_pretrained(self.config["model_id"])
        model = AutoModelForZeroShotObjectDetection.from_pretrained(self.config["model_id"])
        return model

    def generateResponse(self, image,text_labels):
        inputs = self.processor(images=image, text=text_labels, return_tensors="pt").to(self.DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )
        return results

if __name__ == "__main__":
    import requests
    import torch
    from PIL import Image

    test = test("test")
    test.loadModel()
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    #requests.get(image_url, stream=True).raw
    image = Image.open("image/AMBER_2.jpg")
    # Check for cats and remote controls
    text_labels = [["a cat", "a remote control"]]

    result = test.generateResponse(image,text_labels)[0]
    for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
        box = [round(x, 2) for x in box.tolist()]
        print(f"Detected {labels} with confidence {round(score.item(), 3)} at location {box}")