from dotenv import load_dotenv
import os
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import torch

class WrapTRL:
    def __init__(
            self, 
            args, 
            **kwargs
            ):
        self.args = args

    def load(self):
        self.device = self.args["device"]
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args["model_id"],
            device_map = "auto"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args["model_id"],
            device_map = "auto"
            )
        self.train_dataset = load_dataset(
            self.args["dataset_name"],
            split = "train"
        )

    def setup_training(self):
        if self.args["training_method"] == "sft":
            self.training_args = SFTConfig(
                packing=True,
                num_train_epochs=self.args["num_train_epochs"],
                output_dir=self.args["output_dir"],   
            )
            self.trainer = SFTTrainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.train_dataset,
            )
        else:
            raise NotImplementedError(f"Training method {self.args['training_method']} not implemented.")

    def train(self):
        self.model.train()
        self.trainer.train()

    def eval(self):
        self.model.eval()
        gsm8k = load_dataset("openai/gsm8k", "main", split="test")

        # Evaluate on the first 5 samples
        for i in range(5):
            sample = gsm8k[i]
            prompt = sample["question"]

            # Tokenize and prepare input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            # Generate output
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    top_p=0.95,
                    temperature=0.7
                )

            # Decode and print result
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"\n=== Sample {i+1} ===")
            print(f"Question: {prompt}\n")
            print(f"Model Response: {response}\n")
            print(f"Reference Answer: {sample['answer']}\n")
    
    def get_response(self, prompt):

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                top_p=0.95,
                temperature=0.7
            )

        # Decode and print result
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

if __name__ == "__main__":
    # Load environment and login to Weights & Biases
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    # Write debug code here