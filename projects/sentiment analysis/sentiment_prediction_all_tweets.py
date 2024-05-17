import pandas as pd
from tqdm import tqdm
import time

from llama_cpp import Llama

def generate_prompt(all_tweets, debug=False):
    '''Create an expert linguist prompt with 1-shot tweet examples'''

    # expert linguist prompt
    system_prompt = f'''You are an expert at linguistic analysis \
and your area of expertise is interpreting the sentiment of \
texts from tweets published online. You are very intelligent and \
are extremely good at your job. 

You task will go as follows: you will be provided \
with some information containing the text from a tweet.\
You will analyze it expertly and respond with 2 pieces of information.

First, the sentiment of the tweet. It could be \
one of three options: [positive, negative, neutral].\

Secondly, you will point out the words \
from the original tweet that supports that sentiment inference \
in your expert and accurate analysis.'''

    system_prompt = system_prompt + "\n\n3 examples, one from each class are given below:-\n"

    # get 1-shot example tweets from each sentiment class 
    positive_sample = all_tweets[all_tweets.sentiment == "positive"].sample(1).values
    if debug: print(positive_sample, end='\n\n')

    negative_sample = all_tweets[all_tweets.sentiment == "negative"].sample(1).values
    if debug: print(negative_sample, end='\n\n')

    neutral_sample = all_tweets[all_tweets.sentiment == "neutral"].sample(1).values
    if debug: print(neutral_sample)

    n_shot_examples = [positive_sample, negative_sample, neutral_sample]

    for sample in n_shot_examples:
        sample_tweet_content, sample_tweet_sentiment, _ = extract_info(sample)

        one_shot_example = f'''\n{sample_tweet_content}\n\n{sample_tweet_sentiment}\n'''
    
        system_prompt = system_prompt + one_shot_example + "\n"

    system_prompt = system_prompt + "\nNow, you are going to be given information for a single tweet.\n"

    return system_prompt

def extract_info(tweet_arr):
    '''get tweet content, sentiment, and id'''

    tweet_id = tweet_arr[0][0]
    tweet_txt = tweet_arr[0][1]
    tweet_info = f'''Tweet Content: {tweet_txt}'''

    sentiment = tweet_arr[0][3]
    tweet_sentiment = f'''Sentiment: [{sentiment}]'''

    return tweet_info, tweet_sentiment, tweet_id

def get_prediction(prompt):
    '''get last prediction made in the prompt'''
    return '[' + str(prompt.split('[')[-1].split(']')[0]) + ']'

def make_prediction(model, input, prompt_template, debug=False):
    tweet_content, tweet_sentiment, id = extract_info(input)

    question_prompt = prompt_template + tweet_content + \
    '''\nPredict the sentiment class from positive/negative/neutral.
    Do not continue after making the prediction for this single tweet.\n
    Sentiment for '''

    output = llm(
        question_prompt, # Prompt
        max_tokens= 50, # Generate up to [max_tokens] tokens, set to None to generate up to the end of the context window
        echo=True, # Echo the prompt back in the output
        temperature=0, # The temperature parameter (higher for more "creative" or "out of distribution" output)
    )    

    full_output = output['choices'][0]['text']
    prediction_class = get_prediction(full_output)
    true_sentiment = get_prediction(tweet_sentiment)
    prediction_accuracy = 1 if prediction_class == true_sentiment else 0

    pred_dict = {
        'tweet_id': id,
        'tweet_content': tweet_content,
        'tweet_sentiment': true_sentiment,
        'predicted_sentiment': prediction_class,
        'prediction_accuracy': prediction_accuracy,
        'full_output': full_output
        }

    return pred_dict

if __name__ == "__main__":

    # load tweets sentiment data
    # dataset credit: https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset?select=train.csv
    all_tweets = pd.read_csv('data/train.csv', encoding='unicode_escape')

    # system prompt
    prompt_template = generate_prompt(all_tweets)

    # model credit: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF
    mpath = r'C:/Users/chinm/.cache/lm-studio/models/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q5_K_M.gguf'
    
    # load ggup model
    llm = Llama(
        model_path=mpath, # model path
        system_prompt=prompt_template, # system prompt
        n_ctx=2048, # model context
        n_gpu_layers=-1 # numper of layers to offload for GPU acceleration 
        )

    # run inference for N tweets 
    N = 300
    pred_list = []

    print("Starting prediction for {N} tweets...")
    tick = time.monotonic()

    for i in tqdm(range(N)):
        
        test_sample = all_tweets.sample(1).values

        sample_pred_dict = make_prediction(llm, test_sample, prompt_template)

        pred_list.append(sample_pred_dict)

    tock = time.monotonic()
    
    print(f"Completed sentiment analysis for {N} tweets!")
    print(f"Time taken: {(tock-tick):.4f} seconds")
    print(f"Time taken per tweet: {((tock-tick)/N):.4f} seconds")

    output_df = pd.DataFrame(pred_list)
    correct_predictions_count = output_df[output_df.prediction_accuracy == 1].shape[0]
    print(f"Prediction accuracy: {100.0*correct_predictions_count/N}%")

    output_df.to_csv('data/output_df.csv', index=False)
    