# 🚀 WrapTRL: Simplify Your LLM Fine-Tuning Journey

*WrapTRL* is a Python package that streamlines the process of fine-tuning large language models (LLMs) using Hugging Face's [Transformers](https://huggingface.co/transformers/) and [TRL](https://github.com/huggingface/trl) libraries. Whether you're a researcher or developer, WrapTRL offers a concise and flexible interface for supervised fine-tuning (SFT), model evaluation, and inference.

---

## 🎯 Features

* **Plug-and-Play Training**: Easily fine-tune models like `Qwen/Qwen2-0.5B` with minimal setup.
* **Flexible Configuration**: Customize training parameters through a simple `args` dictionary.
* **Integrated Evaluation**: Evaluate model performance on datasets like GSM8K.
* **Inference Ready**: Generate responses to prompts using the `get_response()` method.
* **W\&B Integration**: Monitor training progress with [Weights & Biases](https://wandb.ai/).

---

## 🛠️ Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/wraptrl.git
   cd wraptrl
   ```



2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```



3. **Set Up Environment Variables**:

   Create a `.env` file in the root directory and add your Weights & Biases API key:

   ```env
   WANDB_API_KEY=your_wandb_api_key
   ```



---

## 🚀 Quick Start

```python
from wraptrl import WrapTRL

args = {
    "model_id": "Qwen/Qwen2-0.5B",
    "device": "cuda:0",
    "dataset_name": "openai/gsm8k",
    "num_train_epochs": 3,
    "output_dir": "./wraptrl_output",
    "training_method": "sft"
}

trainer = WrapTRL(args)
trainer.load()
trainer.setup_training()
trainer.train()
trainer.eval()
```



---

## 🧠 Inference Example

```python
response = trainer.get_response("What is the capital of France?")
print(response)
```


---

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request. For major changes, open an issue first to discuss your ideas.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

For questions or suggestions, feel free to open an issue or contact [Ketan](mailto:ketansuhaas@gmail.com).

