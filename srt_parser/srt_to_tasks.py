#!/usr/bin/env python3
# 1. load list of cefr wordlists from A1 to C2
from inspect import cleandoc

import pandas as pd
import argparse, re
from huggingface_hub import InferenceClient

def form_wordlist():

    words = pd.read_csv('wordlist.csv')

    template = """
    {WORD}: {LEVEL}
    """.lstrip()

    wordlist = ''.join([template.format_map(row) for ix, row in words.iterrows()])

    return wordlist.replace('\n', '')
# 2. pre-process srt: line per line

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

    return '\n'.join([x for x in cleaned_lines if len(x) > 0])

# 3. prompt tune LLM to classify word in context for each level

def llm_streaming(filepath, data, wordlist):
    model_name = "Qwen/Qwen2.5-72B-Instruct"

    client = InferenceClient(model_name, token='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')

    output = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Follow the instruction:\n"
                                          "1. You do linguistic analysis of the subtitles\n"
                                          "2. Stick to the provided CEFR-based wordlist:\n"
                                          "==========="
                                          f"{wordlist}"
                                          "==========="
                                          "3.A Find idioms and classify their CEFR level\n"
                                          "3.B Find phrasal verbs and classify their CEFR level\n"
                                          "3.C Find semantically significant words and classify their CEFR level\n" 
                                          "4.D Find collocation and classify their CEFR levels\n"
                                          "5. Do not provide explanations\n"
                                          "6. Return dictionary with the following structure:\n"
                                          "```\n"
                                          "IDIOMS\n: [LIST OF DICTIONARIES {IDIOM : CEFR LEVEL}]\n" 
                                          "VERBS\n: [LIST OF DICTIONARIES {PHRASAL VERB : CEFR LEVEL}]\n"
                                          "WORDS\n: [LIST OF DICTIONARIES {WORD : CEFR LEVEL}}\n"
                                          "COLLOCATIONS\n: [LIST OF DICTIONARIES {COLLOCATION : CEFR LEVEL}]\n"
                                          "```"},
            {"role": "user",
             "content": f"Analyze the following data:\n\n {data}"},
        ],
        stream=True,
        max_tokens=2048,
    )

    llm_response = ''.join([chunk.choices[0].delta.content for chunk in output])

    with open(f'{filepath}_wordlist_.txt', 'a') as f:
        f.write(llm_response)

    return llm_response

def form_batches(lemmas, batch_size=500):

    i, j = 0, batch_size

    batches = list()

    for _ in range(len(lemmas)//batch_size):
        batches.append([elem for elem in lemmas[i:j]])
        i += batch_size
        j += batch_size

    return batches

# 4. load linees and classify each word or phrase
# 5. form csv {unit | type (lemma, phrase verb, idiom, collocation), level}

def main(filepath):
    
    with open(f'{filepath}_wordlist_.txt', 'w') as f:
        f.write('')

    wordlist = form_wordlist()

    data = process_srt(filepath)

    batches = form_batches(data)

    for batch in batches:
        llm_streaming(filepath, batch, wordlist)

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="*.srt parser for NLP, takes filepath to *.srt as input")

    # Add arguments
    parser.add_argument("filepath", type=str, help="A path to *.srt file")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.filepath)