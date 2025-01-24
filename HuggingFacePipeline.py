import json
from abc import ABC, abstractmethod

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.set_default_device(DEVICE)

class AbstractModel(ABC):
    def __init__(self, model_name:str):
        self.model_alias = model_name
        self.config= self.load_model_config()
        self.id=self.config['model_id']
        self.model_type=self.config['model_type']
        self.DEVICE = self.cuda()


    def load_model_config(self):
        file_name = f'./configs/alias_{self.model_alias}.json'
        with open(file_name, 'r', encoding='utf-8') as cfile:
            data = json.load(cfile)
        return data

    @staticmethod
    def cuda():
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        if DEVICE == 'cuda':
            print("Using GPU")
            torch.set_default_dtype(torch.bfloat16)
        torch.set_default_device(DEVICE)
        return DEVICE


    @abstractmethod
    def generateResponse(self, **inputs):
        pass
    @abstractmethod
    def generateModel(self, **inputs):
        pass

