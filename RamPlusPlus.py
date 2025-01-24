import torch
from ram import get_transform
from ram import inference_ram as inference
from ram.models import ram_plus

import HuggingFacePipeline


class RamPlus(HuggingFacePipeline.AbstractModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.image_size = self.config['imageSize']
        self.transform = get_transform(self.image_size)
        self.model = self.generateModel()
        self.model.eval()

    def generateModel(self, **inputs):
        model = ram_plus(pretrained=self.config['pretrained'],model_type=self.config['model_type'])
        model.to(self.DEVICE)
        return model

    def generateResponse(self, images):
        result=[]
        for image in images:
            image = self.transform(image).unsqueeze(0).to(self.DEVICE)
            output,_ = inference(image, self.model)
            result+=(output.split('|'))

        return result

def test():
    model_type = 'swin_l'
    pretrained = 'pretrained/ram_plus_swin_large_14m.pth'
    from PIL import Image
    imagePath = "./image/AMBER_2.jpg"
    im = Image.open(imagePath)
    im.resize((384, 384))
    ra = RamPlus(384)
    return ra.generateResponse([im])
if __name__ == "__main__":
   print(test())