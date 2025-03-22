import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor

from Models.AbstractModel import AbstractModel


class Florence2(AbstractModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.task_prompt= self.config['task_prompt']
        self.text_prompt= self.config['text_prompt']
        self.prompt= self.task_prompt + self.text_prompt


    def generateModel(self, **inputs):
        model = AutoModelForCausalLM.from_pretrained(self.config["model_id"], torch_dtype=self.torch_dtype, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(self.config["model_id"], trust_remote_code=True)
        return model

    def generateResponse(self, image):
        image=np.array(image)
        inputs = self.processor(text=self.prompt, images=image, return_tensors="pt").to(self.DEVICE, self.torch_dtype)
        generated_ids =self.model.generate(
          input_ids=inputs["input_ids"],
          pixel_values=inputs["pixel_values"],
          max_new_tokens=self.config['max_new_tokens'],
          num_beams=self.config['num_beams']
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        parsed_answer = self.processor.post_process_generation(generated_text, task=self.task_prompt, image_size=(image.shape[1], image.shape[0]))
        return list(parsed_answer.values())

    def cleanModel(self):
        super().cleanModel()
        del self.processor

if __name__ == "__main__":
    import os
    def test():
        from PIL import Image
        imagePath = "S:\phili\GitHub\Hallucination-Free-er-Image-Captions\image\AMBER_2.jpg"
        im = Image.open(imagePath)
        im.resize((384, 384))
        ra = Florence2("Florence2")
        ra.loadModel()
        return ra.generateResponse(im)
    print(test())