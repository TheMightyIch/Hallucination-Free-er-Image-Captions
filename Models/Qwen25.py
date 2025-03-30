import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from Models.AbstractModel import AbstractModel


class Qwen25(AbstractModel):
    def __init__(self, model_name ):
        super().__init__(model_name)
        self.responseLength = int(self.config['responseLength'])
        self.temperature = float(self.config['temperature'])

    def generateModel(self, **inputs):
        model = AutoModelForCausalLM.from_pretrained(
                self.id,device_map=self.DEVICE, torch_dtype="auto", cache_dir=self.config["cache_dir"]
            ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
           self.id, cache_dir=self.config["cache_dir"]
        )
        return model



    def generateResponse(self, prompt):
        messages = [
            {"role": "system", "content": self.config["system_prompt"]},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=self.config["add_generation_prompt"],
        )
        input=self.tokenizer([text], return_tensors="pt").to(self.DEVICE)
        with torch.no_grad():
            output = self.model.generate(**input, max_new_tokens=self.responseLength, temperature=self.temperature)
        generated_text = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(input.input_ids, output)
        ]

        return self.tokenizer.batch_decode(generated_text, skip_special_tokens=True)[0]


    def cleanModel(self):
        super().cleanModel()
        del self.tokenizer


if __name__ == "__main__":
    Qwen_id ="Qwen25"
    Qwen25_7B="Qwen/Qwen2.5-3B-Instruct"
    Qwen25 = Qwen25(Qwen_id)
    Qwen25.loadModel()
    print("Qwen25 loaded")
    prompt = "Give me a short introduction to large language model."
    print(Qwen25.generateResponse(prompt))
