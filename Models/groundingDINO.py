
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
from Models.AbstractModel import AbstractModel


class groundingDINO(AbstractModel):
    def __init__(self,model_name: str):
        super().__init__(model_name)

    def generateModel(self, **inputs):
        self.processor = AutoProcessor.from_pretrained(self.config["model_id"])
        model = AutoModelForZeroShotObjectDetection.from_pretrained(self.config["model_id"])
        return model

    def generateResponse(self, image: Image,text):
        try:
            inputs = self.processor(images=image, text=text, is_split_into_words=self.config["tokenized"], return_tensors="pt").to(self.DEVICE)
        except Exception as e:
            print(type(image))
            print(image.filename)
            print(e)
            inputs = self.processor(images=image, text=text, is_split_into_words= not self.config["tokenized"], return_tensors="pt").to(self.DEVICE)
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
        del inputs
        return results

    def cleanModel(self):
        super().cleanModel()
        del self.processor

if __name__ == "__main__":
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