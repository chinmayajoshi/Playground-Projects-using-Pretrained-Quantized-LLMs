# Sentiment Analysis Using Quantized GGUF Model

This repository contains a Python script for performing sentiment analysis on tweets using a  quantized GGUF model. The model is loaded and run using the `llama-cpp-python` library.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Description](#description)
- [Model and Dataset](#model-and-dataset)
- [Results](#results)
- [Miscellaneous](#Miscellaneous)

## Installation

1. Clone the repository or download just the `sentiment_prediction_all_tweets.py` script available [here](https://github.com/chinmayajoshi/Playground-Projects-using-Pretrained-Quantized-LLMs/blob/main/projects/sentiment%20analysis/sentiment_prediction_all_tweets.py).

2. Install the llama.cpp python binding package. For more information on the library, check out the documentation [here](https://llama-cpp-python.readthedocs.io/en/latest/).

    ```sh
    pip install llama-cpp-python
    ```

3. Download the dataset and place it in the `data` directory. The dataset can be found [here](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset?select=train.csv).

4. Ensure the GGUF model file is available in the specified path. The model (5-bit Quantized Mistral 7B Instruct) I used can be found [here](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF).

## Usage

To run the sentiment analysis, use the following command:

```sh
python sentiment_prediction_all_tweets.py
```

## Description

The script performs sentiment analysis using the following steps:
1. **Generate System Prompt**: Creates an expert linguist prompt with 1-shot examples for each sentiment class.

```sh
You are an expert at linguistic analysis and your area of expertise is interpreting the sentiment of texts from tweets published online. 
You are very intelligent and are extremely good at your job. 

You task will go as follows: you will be provided with some information containing the text from a tweet.
You will analyze it expertly and respond with 2 pieces of information.

First, the sentiment of the tweet. It could be one of three options: [positive, negative, neutral].
Secondly, you will point out the words from the original tweet that supports that sentiment inference in your expert and accurate analysis.

3 examples, one from each class are given below:-

Tweet Content: <insert-tweet-content-1>
Sentiment: [positive]

Tweet Content: <insert-tweet-content-2>
Sentiment: [negative]

Tweet Content: <insert-tweet-content-3>
Sentiment: [neutral]

Now, you are going to be given information for a single tweet.
Tweet Content: <insert-input-tweet-content>

Predict the sentiment class from positive/negative/neutral.
Do not continue after making the prediction for this single tweet.

Sentiment for
```

2. **Model Loading**: Loads the GGUF model using the `llama-cpp-python` library.
3. **Inference**: Runs inference on a sample of tweets to predict their sentiment and identify supporting words.
4. **Results**: Calculates the prediction accuracy and saves the results.

## Model and Dataset

- **Model**: [Mistral-7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)
- **Dataset**: [Sentiment Analysis Dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset?select=train.csv)

## Results

The script will output the sentiment analysis results to `data/output_df.csv` and print the overall accuracy.

```sh
Completed sentiment analysis for 300 tweets!
Time taken: 3410.8910 seconds
Time taken per tweet: 11.3696 seconds
Prediction accuracy: 56.0%
```

## Miscellaneous (Parting Thoughts)

1. <u>**Inference Runtime:**</u><br>
Wondering why the performance is so slow? <br>
For eg. I ran the above toy benchmark on just 300 tweets even though the dataset coutains over 27k labelled tweets. With `~10 secs/tweet` it would take `7.5 hours` of inference time.<br><br>
Okay. So how can we fix this?<br>
The script ran purely on CPU but GPU acceleration is supported. It can be used to offload model layers by changing the `n_gpu_layers` parameter which I did not use for the above toy benchmark. 
<br>TL;DR- Use a (better) GPU.

2. <u>**Sentiment Prediction Accuracy:**</u><br>
56% accuracy on a 3-class sentiment analysis is great, if you are still using hand-coded features in the 80s. Even a simple linear classifier or a decision tree achieves better performance than this.<br><br>
Okay. So how do we improve that? 
<br>Two suggestion: Use better prompt engineering and a better LLM model.