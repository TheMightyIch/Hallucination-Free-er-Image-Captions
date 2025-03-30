import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from Models.AbstractModel import AbstractModel


class LLAMA32(AbstractModel):
    def __init__(self, model_name ):
        super().__init__(model_name)
        self.token = os.environ["HFACESSTOKEN"]
        self.responseLength = int(self.config['responseLength'])
        self.temperature = float(self.config['temperature'])
        self.Quantization = self.config['Quantization'] if self.config['Quantization'] else ""

    def generateModel(self, **inputs):
        if self.Quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.id,token= self.token,device_map="auto", cache_dir=self.config["cache_dir"],
                quantization_config=quantization_config).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                    self.id,token= self.token,device_map="auto", cache_dir=self.config["cache_dir"]
                ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
           self.id,token= self.token, cache_dir=self.config["cache_dir"]
        )
        self.tokenizer.additional_special_tokens.append({"pad_token":"<pad>"})
        model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        return model



    def generateResponse(self, messages):
        input=self.tokenizer.encode(messages, return_tensors="pt").to(self.DEVICE)
        with torch.no_grad():
            output = self.model.generate(input,pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=self.responseLength, temperature=self.temperature)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def cleanModel(self):
        super().cleanModel()
        del self.tokenizer


if __name__ == "__main__":
    llama_id ="LLAMA32"
    llama3_8B="meta-llama/Meta-Llama-3-8B-Instruct"
    task = "text-generation"
    device ="auto"
    Llama = LLAMA32(llama_id)
    Llama.loadModel()
    print(Llama.generateResponse("Hello, how are you?"))
