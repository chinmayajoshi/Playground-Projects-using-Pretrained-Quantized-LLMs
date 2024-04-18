# Playground for Projects using Pretrained Quantized LLMs
A place for me to run quantized LLMs locally and testing their capabilities with toy projects.<br>
Not a place for serious projects, just for fun.

## Getting Started
All notebooks and scripts in this repo use the [Python binding](https://github.com/abetlen/llama-cpp-python) for [llama.cpp](https://github.com/ggerganov/llama.cpp) to run .gguf LLMs.<br>

> What are .gguf models?

GGUF is a file extension for Quantized Model files which converts FP32/FP16 bit model weights into 8bit/5bit/4bit/etc 


> Why use .gguf models?

Quantized models have lower bit model weights, which require less compute for inference (at some output quality loss). So they can use [CPU and RAM] instead of [GPU and VRAM] for model inference. 

GPUs are expensive and fast. CPUs are cheap and slow.<br>
Hence, less tokens/second speed for model inference, but at a lower hardware requirement (eg. very useful at edge inference).

> How are we using these models in Python?

Check out the [demo notebook](https://github.com/chinmayajoshi/Playground-Projects-using-Pretrained-Quantized-LLMs/blob/main/demo_loading_gguf_model.ipynb) for sample inference of 5-bit Quantized Mistral 7B Instruct Model from [TheBloke from HuggingFace](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF).


## List of Projects
TODO