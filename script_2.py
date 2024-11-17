#!/usr/bin/env python3
import argparse, json, re
from traceback import print_tb

from huggingface_hub import InferenceClient

import pandas as pd

def llm_streaming(lemmas_batch):
    model_name = "Qwen/Qwen2.5-72B-Instruct"
    client = InferenceClient(model_name, token='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')
    output = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "you are english teacher assistant\n"
                                          "generate 10 diverse tasks to learn the provided words\n"
                                          "return dictionary\n"
                                          "choose only frequent collocations\n"
                                          "example: {'word': 'word', 'task': 'task description', 'solution': 'right answer']}\n"
                                          "use the following tasks types:\n"
                                          "- fill-in-the-gap;\n"
                                          "- multiple choice;\n"
                                          "- write one-word translation from English to Russian;\n"
                                          "- write one-word translation from Russian to English;\n"
                                          "- choose a correct translation into Russian"
             },
            {"role": "user",
             "content": f"Analyze the following list and return dictionary with task descriptions and solutions: \n\n {lemmas_batch}"},
        ],
        stream=True,
        max_tokens=2048,
        temperature=0.7,
    )

    llm_response = ''.join([chunk.choices[0].delta.content for chunk in output])
    pattern = re.compile(r'\[.*?\]', re.DOTALL)
    # Find all matches
    matches = pattern.findall(llm_response)

    temp = dict()
    temp['word'] = []
    temp['solution'] = []
    temp['task'] = []

    for elem in matches[0].split('\n'):
        elem = elem.strip()
        elem = elem.split("'")

        if len(elem) > 2:
            if elem[1] == 'word':
                temp['word'].append(elem[-2])
            elif elem[1] == 'solution':
                temp['solution'].append(elem[-2])
            elif elem[1] == 'task':
                temp['task'].append(elem[-2])

    return temp

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
    df = pd.read_csv(f"{filepath}_cefr.csv")
    df_lemmas = pd.DataFrame(llm_streaming(df['lemmas']))
    df_phrasal = pd.DataFrame(llm_streaming(df['phrasal verbs']))
    df_idioms = pd.DataFrame(llm_streaming(df['idioms']))
    df_collocations = pd.DataFrame(llm_streaming(df['collocations']))
    df_final = pd.concat([df_lemmas, df_phrasal, df_idioms, df_collocations])
    df_final.to_csv(f'{filepath}_tasks.csv', index=False)

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="*.srt parser for NLP, takes filepath to *.srt as input")

    # Add arguments
    parser.add_argument("filepath", type=str, help="A path to *.srt file")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.filepath)