#!/usr/bin/env python3
import argparse, re
import pandas as pd

import nltk
nltk.download('punkt')

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

def form_wordlist():

    words = pd.read_csv('wordlist.csv')

    template = """
    {WORD}: {LEVEL}
    """.lstrip()

    wordlist = ''.join([template.format_map(row) for ix, row in words.iterrows()])

    return wordlist.replace('\n', '')

#wordlist = form_wordlist()

def main(filepath):
    with open(f'{filepath}.txt', 'w') as f:
        f.write(''.join(process_srt(filepath)))

    #with open('wordlist.txt', 'w') as f:
    #    f.write(''.join(wordlist))

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="*.srt parser for NLP, takes filepath to *.srt as input")

    # Add arguments
    parser.add_argument("filepath", type=str, help="A path to *.srt file")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.filepath)