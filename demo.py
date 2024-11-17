#!/usr/bin/env python3
import argparse, re

from huggingface_hub import InferenceClient

import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

model_name = "Qwen/Qwen2.5-72B-Instruct"
client = InferenceClient(model_name, token='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')

def process_srt(filepath):

    with open(filepath, 'r', encoding='latin1') as f:
        data = f.read()

    cleaned_lines = []

    lines = data.split('\n')

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip index and timestamp lines
        if re.match(r'^\d+$', line) or re.match(r'^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$', line):
            i += 1
            continue

        # Collect subtitle text
        subtitle_text = []
        while i < len(lines) and not (
                re.match(r'^\d+$', lines[i]) or re.match(r'^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$',
                                                         lines[i])):
            subtitle_text.append(lines[i].strip())
            i += 1

        # Join the subtitle text and add it to the cleaned lines
        cleaned_lines.extend(subtitle_text)

    cleaned_text = ' '.join([x for x in cleaned_lines if len(x) > 0])
    cleaned_text = cleaned_text.replace('<i>', '').replace('</i>', '').replace('...', '')
    return re.sub(r"[^a-zA-Z.,:;\?!']", ' ',  cleaned_text)

def main(filepath):
    data = sent_tokenize(process_srt(filepath))

    k = 0
    j = 5
    data_ = []

    for i in range(len(data)//5):
        t = ' '.join(data[k:j])
        k += 5
        j += 5
        data_.append(t)

    for d in data_:
        # word
        # LVL
        # task
        print(data_)

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="*.srt parser for NLP, takes filepath to *.srt as input")

    # Add arguments
    parser.add_argument("filepath", type=str, help="A path to *.srt file")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.filepath)
