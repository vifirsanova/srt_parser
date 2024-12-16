#!/usr/bin/env python3
import argparse, ast
import json
from huggingface_hub import InferenceClient

import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

model_name = "Qwen/Qwen2.5-72B-Instruct"
client = InferenceClient(model_name, token='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')

def get_words(d):
    output = client.chat.completions.create(
        max_tokens=512,
        messages=[
            {
                "role": "system",
                "content": "extract topically important words, all the collocations, "
                           "all the phrasal verbs, idioms from the given text, "
                           "save to the following variables: WORD, COLLOCATION, VERB, IDIOM"
                           "return json",
            },
            {
                "role": "user",
                "content": f"{d}",
            }
        ],
        response_format={
                            "type": "json",
                            "value": {
                                "properties": {
                                    "WORD": {"type": "array", "items": {"type": "string"}},
                                    "COLLOCATION": {"type": "array", "items": {"type": "string"}},
                                    "VERB": {"type": "array", "items": {"type": "string"}},
                                    "IDIOM": {"type": "array", "items": {"type": "string"}}
                                },
                                "required": ["WORD", "COLLOCATION", "VERB", "IDIOM"],
                            },
                        }
    )

    return ast.literal_eval(output.choices[0].message.content)

def main(filepath):
    with open(f'{filepath}.txt', 'r') as f:
        data = sent_tokenize(f.read())

    k = 0
    j = 5
    data_ = []

    for i in range(len(data)//5):
        t = ' '.join(data[k:j])
        k += 5
        j += 5
        data_.append(t)

    print(len(data_))

    words = dict()
    words['WORD'] = []
    words['COLLOCATIONS'] = []
    words['VERB'] = []
    words['IDIOM'] = []

    i = 0
    for d in data_[:10]:
        print(i)
        i+= 1

        temp = get_words(d)
        words['WORD'].append(temp['WORD'])
        words['COLLOCATIONS'].append(temp['COLLOCATION'])
        words['VERB'].append(temp['VERB'])
        words['IDIOM'].append(temp['IDIOM'])

        with open(f'{filepath}_words.json', 'w', encoding='utf-8') as outfile:
                json.dump(words, outfile)

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="*.srt parser for NLP, takes filepath to *.srt as input")

    # Add arguments
    parser.add_argument("filepath", type=str, help="A path to *.srt file")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.filepath)