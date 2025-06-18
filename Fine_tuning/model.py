from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Model:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype = torch.float16,
            device_map = "auto",
            trust_remote_code = True
        )

if __name__ == "__main__":
    model = Model(model_name = "Qwen/Qwen2.5-7B-Instruct", device = "mps")
    print(model.model)