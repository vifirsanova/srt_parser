#!/usr/bin/env python3
import argparse, json, re

from huggingface_hub import InferenceClient

import pandas as pd

def llm_streaming(lemmas_batch):
    model_name = "Qwen/Qwen2.5-72B-Instruct"
    client = InferenceClient(model_name, token='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')
    output = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "you are english teacher assistant\n"
                                          "pick and return list of words from intermediate and advanced vocabulary\n"
                                          "do not provide explanations, do not return misspelled words\n"
                                          "use oxford vocabulary as reference\n"
                                          "example: ['this', 'word']\n"
                                          "do not use word from the given example\n"},
            {"role": "user",
             "content": f"Analyze the following list of verb, and return list of words for B2-C2 English learners: \n\n {lemmas_batch}"},
        ],
        stream=True,
        max_tokens=2048,
    )

    llm_response = ''.join([chunk.choices[0].delta.content for chunk in output])
    llm_response = llm_response.replace('[','').replace(']', '')
    return [x.strip() for x in llm_response.split(',')]

def form_batches(lemmas, batch_size=100):
    i = 0
    j = batch_size
    batch = list()
    for _ in range(len(lemmas)//batch_size):
        batch.append([elem for elem in lemmas[i:j]])
        i += batch_size
        j += batch_size
    return batch

def main(filepath):
    with open(f"{filepath}_lemmas.txt", "r") as file:
        lemmas = file.read()

    with open(f"{filepath}_phrases.json", "r") as json_file:
        phrases = json.load(json_file)

    data = {'lemmas': []}

    lemmas_batch = form_batches(lemmas.split('\n'))

    for batch in lemmas_batch:
        temp = llm_streaming(batch)

        for word in temp:
            data['lemmas'].append(word) # process lemmas

    for key in phrases.keys():
        data[key] = []
        temp = llm_streaming(phrases[key])

        for word in temp:
            data[key].append(word)

        data[key] = data[key] + [None] * (len(data['lemmas']) - len(data[key]))

    df = pd.DataFrame(data)
    df.to_csv(f'{filepath}_cefr.csv', index=False)

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="*.srt parser for NLP, takes filepath to *.srt as input")

    # Add arguments
    parser.add_argument("filepath", type=str, help="A path to *.srt file")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.filepath)