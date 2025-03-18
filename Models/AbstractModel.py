import json
from abc import ABC, abstractmethod
import os
import torch



class AbstractModel(ABC):
    def __init__(self, model_name:str):
        self.model_alias = model_name
        self.config= self.load_model_config()
        self.id=self.config['model_id']
        self.model_type=self.config['model_type']
        self.DEVICE,self.torch_dtype = self.cuda()


    def load_model_config(self):
        file_name = f'{os.getcwd()}/Models/configs/alias_{self.model_alias}.json'
        with open(file_name, 'r', encoding='utf-8') as cfile:
            data = json.load(cfile)
        return data

    @staticmethod
    def cuda():
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        if DEVICE == 'cuda':
            print("Using GPU")
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        torch.set_default_device(DEVICE)
        return DEVICE,torch_dtype

    def loadModel(self, **inputs):
        self.model=self.generateModel(**inputs)
        self.model.eval()
        if self.config["ToDevice"]=="no":
            return
        self.model.to(self.DEVICE)
        print(f"Model {self.model_alias} loaded to {self.DEVICE}")

    @abstractmethod
    def generateResponse(self, **inputs):
        pass
    @abstractmethod
    def generateModel(self, **inputs):
        pass

    @abstractmethod
    def cleanModel(self):
        del self.model
        pass

