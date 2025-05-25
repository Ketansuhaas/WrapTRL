from setuptools import setup, find_packages

setup(
    name="wraptrl",
    version="0.1.0",
    author="Your Name",
    author_email="youremail@example.com",
    description="A simple wrapper for fine-tuning LLMs using Hugging Face Transformers and TRL.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/wraptrl",
    packages=find_packages(),
    install_requires=[
        "datasets==3.1.0",
        "python-dotenv==1.1.0",
        "torch==2.6.0",
        "transformers==4.51.1",
        "trl==0.16.1",
        "wandb==0.19.11"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
