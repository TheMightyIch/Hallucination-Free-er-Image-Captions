import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    print("Using GPU")
    torch.set_default_dtype(torch.bfloat16)
torch.set_default_device(DEVICE)

class LLMHuggingFace:
    def __init__(self, model_id,**kwargs,):
        self.token= os.environ["HFACESSTOKEN"]
        self.model_name = model_id
        self.responseLength = 100
        if "Quantization" in kwargs and kwargs["Quantization"] == "8bit":
            self.quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,token= self.token,device_map="auto", cache_dir=kwargs["cache_dir"],
                quantization_config=self.quantization_config).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,token= self.token,device_map="auto", cache_dir=kwargs["cache_dir"]
                ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
           self.model_name,token= self.token, cache_dir=kwargs["cache_dir"]
        )
        self.tokenizer.additional_special_tokens.append({"pad_token":"<pad>"})
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def generateResponse(self, messages):
        #self.model.to(DEVICE)
        self.model.eval()
        input=self.tokenizer.encode(messages, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(input,pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=int (self.responseLength))
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    llama_id ="meta-llama/Llama-3.2-3B-Instruct"
    llama3_8B="meta-llama/Meta-Llama-3-8B-Instruct"
    task = "text-generation"
    device ="auto"
    kwargs= {"cache_dir":"./cache", "Quantization":"8bit"}
    Llama = LLMHuggingFace( llama_id, **kwargs)
    print(Llama.generateResponse("Hello, how are you?"))
