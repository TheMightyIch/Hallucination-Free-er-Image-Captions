from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from Models.AbstractModel import AbstractModel


class groundingDINO(AbstractModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)

    def generateModel(self, **inputs):
        self.processor = AutoProcessor.from_pretrained(self.config["model_id"])
        model = AutoModelForZeroShotObjectDetection.from_pretrained(self.config["model_id"])
        return model

    def generateResponse(self, image,text):
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.DEVICE)
        outputs =self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.config["box_threshold"],
            text_threshold=self.config["text_threshold"],
            target_sizes=[image.size[::-1]]
        )
        return results
