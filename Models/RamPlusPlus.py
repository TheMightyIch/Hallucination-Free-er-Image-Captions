import os

import torch
from ram import get_transform
from ram import inference_ram as inference
from ram.models import ram_plus

from Models.AbstractModel import  AbstractModel


class RamPlusPlus(AbstractModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.image_size = self.config['imageSize']
        self.transform = get_transform(self.image_size)

    def generateModel(self, **inputs):
        model = ram_plus(pretrained=self.config['model_id'],vit=self.config['model_type'],image_size=self.image_size)
        return model

    def generateResponse(self, image):
        image = self.transform(image).unsqueeze(0).to(self.DEVICE)
        output,_ = inference(image, self.model)
        return output.split('|')

def test():
    model_type = 'swin_l'
    pretrained = 'pretrained/ram_plus_swin_large_14m.pth'
    from PIL import Image

    imagePath = os.path.join(os.path.dirname(os.getcwd())+"/image/AMBER_2.jpg")
    im = Image.open(imagePath)
    im.resize((384, 384))
    ra = RamPlusPlus("RamPlusPlus")
    ra.loadModel()
    return ra.generateResponse(im)
if __name__ == "__main__":
   print(test())