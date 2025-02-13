from urllib.request import urlopen
from PIL import Image
import torch
from timm.data.transforms_factory import create_transform
import timm

from Models.AbstractModel import AbstractModel


img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

class EVA(AbstractModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        data_config = self.model.default_cfg
        self.transforms = create_transform(**data_config, is_training=False)

    def generateModel(self, **inputs):
        return timm.create_model(self.config["model_id"], pretrained=True)

    def generateResponse(self, image):
        with torch.no_grad():
            output = self.model(self.transforms(image).unsqueeze(0))
        return output


# model = timm.create_model('eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=True)
# model = model.eval()

if __name__ == "__main__":
    "hf_hub:timm/eva02_base_patch14_448.mim_in22k_ft_in1k"
    eva = EVA("eva02_large_patch14_448.mim_m38m_ft_in22k_in1k")
    print(torch.topk(eva.generateResponse(img).softmax(dim=1) * 100, k=5))
