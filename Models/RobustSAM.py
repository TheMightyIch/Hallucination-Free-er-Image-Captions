import torch
from transformers import AutoProcessor, AutoModelForMaskGeneration

from Models.AbstractModel import  AbstractModel

class RobustSAM(AbstractModel):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.processor = AutoProcessor.from_pretrained(self.id)

    def generateModel(self, **inputs):
        model = AutoModelForMaskGeneration.from_pretrained(self.id, cache_dir=self.config["cache_dir"])
        return model

    def generateResponse(self, image):
        inputs = self.processor(image,input_points=[[[450, 600]]], return_tensors="pt").to("cuda")

        # generate masks using the model
        with torch.no_grad():
            outputs = self.model(**inputs)
        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                                                             inputs["reshaped_input_sizes"].cpu())
        return outputs, masks

    def cleanModel(self):
        super().cleanModel()
        del self.processor



if __name__ == "__main__":
    from PIL import Image
    import requests
    import numpy as np
    import matplotlib.pyplot as plt
    img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    sam = RobustSAM("RobustSAM")
    sam.loadModel()
    outputs,masks=sam.generateResponse(raw_image)

    print(masks)

    def show_mask(mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

        # get the height and width from the mask
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1)
        ax.imshow(mask_image)


    # display the original image
    plt.imshow(np.array(raw_image))
    ax = plt.gca()

    # loop through the masks and display each one
    for mask in masks:
        show_mask(mask, ax=ax, random_color=True)

    plt.axis("off")

    # show the image with the masks
    plt.show()

