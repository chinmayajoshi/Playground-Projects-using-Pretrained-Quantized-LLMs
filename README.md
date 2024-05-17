# Playground for Projects using Pretrained Quantized LLMs
A place for me to run quantized LLMs locally and testing their capabilities with toy projects.<br>
Not a place for serious projects, just for fun.

## Getting Started
All notebooks and scripts in this repo use the [Python binding](https://github.com/abetlen/llama-cpp-python) for [llama.cpp](https://github.com/ggerganov/llama.cpp) to run .gguf LLMs.<br>

> What are .gguf models?

GGUF is a file extension for Quantized Model files which converts FP32/FP16/BF16 bit model weights into 8bit/5bit/4bit/etc 


> Why use .gguf models?

Quantized models have lower bit model weights, which require less compute for inference (at some output quality loss). So they can use [CPU and RAM] instead of [GPU and VRAM] for model inference. 

GPUs are expensive and fast. CPUs are cheap and slow.<br>
Hence, less tokens/second speed for model inference, but at a lower hardware requirement (eg. very useful at edge inference).

> How are we using these models in Python?

Check out the [demo notebook](https://github.com/chinmayajoshi/Playground-Projects-using-Pretrained-Quantized-LLMs/blob/main/demo_loading_gguf_model.ipynb) for sample inference of 5-bit Quantized Mistral 7B Instruct Model from [TheBloke from HuggingFace](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF).


## List of Projects

All projects present in the [/projects](https://github.com/chinmayajoshi/Playground-Projects-using-Pretrained-Quantized-LLMs/tree/main/projects) directory.

1. <u>_Interpretable Sentiment Analysis_ :</u> 

Interpretable Sentiment Analysis on Tweets using 5-bit Mistral 7b Instruct Model.<br>
Checkout the [notebook](https://github.com/chinmayajoshi/Playground-Projects-using-Pretrained-Quantized-LLMs/blob/main/projects/sentiment%20analysis/predicting_sentiment.ipynb) here. Work in progress.

TODO- run for all tweets<br>
TODO- compare llm output with true labels

 2. <u>_Image Classification using LLM_:</u> 

CIFAR-10 Image conversion to ASCII Text, used as an input for the LLM.<br>Tried to check the ASCII text quality for images as an input to the LLM. Stopped after seeing the ASCII text for now [notebook](https://github.com/chinmayajoshi/Playground-Projects-using-Pretrained-Quantized-LLMs/blob/main/projects/image2ascii/image2ascii.ipynb). Might continue later.