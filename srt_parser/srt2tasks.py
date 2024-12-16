#!/usr/bin/env python3
import argparse, ast, json
from huggingface_hub import InferenceClient
from pprint import pprint

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

def main(filepath):
    # todo:
    # 1. берем сабы, чистим
    # 2. делим по новой строке
    # 3. контатенируем по 10 строк
    # 4. делаем оверлэп по 5
    with open(filepath, 'r') as f:
        source = f.read()

    response_format = {
        "type": "json",
        "value": {
            "properties": {
                "word": {"type": "string"},
                #"translation_Russian": {"type": "string"},
                "task": {"type": "string"},
                "solution": {"type": "string"}
            },
            "required": ["word", "task", "solution"],
        },
    }

    tasks = [{'task': 'multiple choice, choose correct spelling',
             'reference': '''
             {'solution': '_____',
                     'task': "Choose the correct correct spelling for the word '______':\n"
                     '1. ______.\n'
                     '2. ______.\n'
                     '3. ______.\n'
                     '4. ______,
                     'word': '_____'}
                     '''},
             {'task': 'multiple choice, choose translation into Russianfill-in-the-gap',
              'reference': '''
                 {'solution': '_____',
                         'task': "Choose the correct correct translation for the word: '______':\n"
                         '1. ______.\n'
                         '2. ______.\n'
                         '3. ______.\n'
                         '4. ______,
                         'word': '_____'}
                         '''},
             ]
             #'correct the spelling; write the correct spelling',
             #'multiple choice; fill-in-the gap',
             #'multiple choice, choose correct definition',
             #'multiple choice; translate into Russian',
             #'write down translation for the word into Russian']

    model_name = "Qwen/Qwen2.5-72B-Instruct"
    client = InferenceClient(model_name, token='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')

    for task in tasks:
        t, r = task.values()
        instruction = f"""
        1. extract a word, collocation, phrasal verb, or idiom from the given text for B1-B2 Intermediate Plateau English learners => save it to "word""
        2. generate one task for English learners, follow the task type: {t} => save to "tasks"
        4. provide a solution to the task => save the solution to "solutions"
        """

        output = client.chat.completions.create
            max_tokens=2048,
            response_format=response_format,
            messages=[
                {
                    "role": "system", "content": "you are helpful English teacher assistant\n"
                                                 "your task is to return *.json based on the provided instructions\n"
                                                 f"use the data from the provided source: {source}\n"
                                                 f"follow the instructions: {instruction}\n"
                                                 "Refer to CEFR Oxford 3000 wordlist\n"
                                                 f"FOLLOW THE REFERENCE: {r}"
                }
            ]
        )
        o = ast.literal_eval(output.choices[0].message.content)
        pprint(o)


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="*.srt parser for NLP, takes filepath to *.srt as input")

    # Add arguments
    parser.add_argument("filepath", type=str, help="A path to *.srt file")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.filepath)