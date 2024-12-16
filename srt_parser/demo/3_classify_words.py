#!/usr/bin/env python3
import argparse, re, ast
import json

import pandas as pd
from huggingface_hub import InferenceClient

import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

model_name = "Qwen/Qwen2.5-72B-Instruct"
client = InferenceClient(model_name, token='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')

def classify_words(current_list, wordlist):
    output = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Return CEFR label for the provided input.\n"
                                          "Stick to the provided CEFR-based wordlist:\n"
                                          f"{wordlist}"
                                          "==========="
                                          "Return dictionary with the following structure:\n"
                                          "WORD: the provided input\n" 
                                          "CEFR: relevant cefr label\n"
                                          "Note! Use CEFR labels from the list: A1, A2, B1, B2, C1, C2\n"
                                          "Do not change the input!"},
            {"role": "user",
             "content": f"Analyze the following word:\n\n {current_list}"},
        ],
        stream=True,
        max_tokens=128,
        response_format={
                            "type": "json",
                            "value": {
                                "properties": {
                                    "WORD": {"type": "string"},
                                    "CEFR": {"type": "string"},
                                },
                                "required": ["WORD", "CEFR"],
                            },
                        }
    )

    llm_response = ''.join([chunk.choices[0].delta.content for chunk in output])

    return ast.literal_eval(llm_response)

def main(filepath):
    with open(f'{filepath}_words.json', 'r') as f:
        data = json.load(f)

    print(data)
    base_dictionary = {
        'type': [],
        'item': [],
        'cefr': [],
        'task': []
    }

    for d in data_:
        temp = get_words(d)
        colocations = [classify_words(items, wordlist) for items in temp['COLLOCATION'] if len(items) > 0]
        idioms = [classify_words(items, wordlist) for items in temp['IDIOM'] if len(items) > 0]
        verbs = [classify_words(items, wordlist) for items in temp['VERB'] if len(items) > 0]
        words = [classify_words(items, wordlist) for items in temp['WORD'] if len(items) > 0]

        for item in colocations:
            tasks = gen_tasks(item)
            base_dictionary['type'].append('collocation')
            base_dictionary['item'].append(item['WORD'])
            base_dictionary['cefr'].append(item['CEFR'])
            base_dictionary['translation'].append(tasks['translation'])
            base_dictionary['spelling_task'].append(tasks['tasks'][0])
            base_dictionary['fill_gap_task'].append(tasks['tasks'][1])
            base_dictionary['multiple_choice_task'].append(tasks['tasks'][2])
            base_dictionary['spelling_sol'].append(tasks['tasks'][0])
            base_dictionary['fill_gap_sol'].append(tasks['solutions'][1])
            base_dictionary['multiple_choice_sol'].append(tasks['solutions'][2])
            df = pd.DataFrame(base_dictionary)
            df.to_csv('data.csv')

        for item in idioms:
            tasks = gen_tasks(item)
            base_dictionary['type'].append('idiom')
            base_dictionary['item'].append(item['WORD'])
            base_dictionary['cefr'].append(item['CEFR'])
            base_dictionary['translation'].append(tasks['translation'])
            base_dictionary['spelling_task'].append(tasks['tasks'][0])
            base_dictionary['fill_gap_task'].append(tasks['tasks'][1])
            base_dictionary['multiple_choice_task'].append(tasks['tasks'][2])
            base_dictionary['spelling_sol'].append(tasks['tasks'][0])
            base_dictionary['fill_gap_sol'].append(tasks['solutions'][1])
            base_dictionary['multiple_choice_sol'].append(tasks['solutions'][2])
            df = pd.DataFrame(base_dictionary)
            df.to_csv('data.csv')

        for item in verbs:
            tasks = gen_tasks(item)
            base_dictionary['type'].append('verb')
            base_dictionary['item'].append(item['WORD'])
            base_dictionary['cefr'].append(item['CEFR'])
            base_dictionary['translation'].append(tasks['translation'])
            base_dictionary['spelling_task'].append(tasks['tasks'][0])
            base_dictionary['fill_gap_task'].append(tasks['tasks'][1])
            base_dictionary['multiple_choice_task'].append(tasks['tasks'][2])
            base_dictionary['spelling_sol'].append(tasks['tasks'][0])
            base_dictionary['fill_gap_sol'].append(tasks['solutions'][1])
            base_dictionary['multiple_choice_sol'].append(tasks['solutions'][2])
            df = pd.DataFrame(base_dictionary)
            df.to_csv('data.csv')

        for item in words:
            tasks = gen_tasks(item)
            base_dictionary['type'].append('word')
            base_dictionary['item'].append(item['WORD'])
            base_dictionary['cefr'].append(item['CEFR'])
            base_dictionary['translation'].append(tasks['translation'])
            base_dictionary['spelling_task'].append(tasks['tasks'][0])
            base_dictionary['fill_gap_task'].append(tasks['tasks'][1])
            base_dictionary['multiple_choice_task'].append(tasks['tasks'][2])
            base_dictionary['spelling_sol'].append(tasks['tasks'][0])
            base_dictionary['fill_gap_sol'].append(tasks['solutions'][1])
            base_dictionary['multiple_choice_sol'].append(tasks['solutions'][2])
            df = pd.DataFrame(base_dictionary)
            df.to_csv('data.csv')

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="*.srt parser for NLP, takes filepath to *.srt as input")

    # Add arguments
    parser.add_argument("filepath", type=str, help="A path to *.srt file")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.filepath)