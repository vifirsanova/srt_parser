#!/usr/bin/env python3
import argparse, re, ast
import pandas as pd
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

def form_wordlist():

    words = pd.read_csv('wordlist.csv')

    template = """
    {WORD}: {LEVEL}
    """.lstrip()

    wordlist = ''.join([template.format_map(row) for ix, row in words.iterrows()])

    return wordlist.replace('\n', '')

wordlist = form_wordlist()

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
    print(ast.literal_eval(llm_response))
    return ast.literal_eval(llm_response)

def gen_tasks(item):
    if item['CEFR'] == 'A1' or item['CEFR'] == 'A2': #or item['CEFR'] == 'B1':
        return 'NaN'
    else:
        output = client.chat.completions.create(
            max_tokens=1024,
            response_format={
                "type": "json",
                "value": {
                    "properties": {
                        "translation": {"type": "string"},
                        "tasks": {"type": "array", "item": {"type": "string"}},
                        "solutions": {"type": "array", "item": {"type": "string"}},
                    },
                    "required": ["translation", "tasks", "solutions"],
                    },
                },
            messages=[
                {
                    "role": "system",
                    "content": "You are given a lexical item in English that a user should learn before watching a movie\n"
                               "1. translate the input into Russian and save the result as 'translation'\n"
                               "2. generate the task that would help to learn the spelling of this lexical item; use the following structure:\n"
                               "STRUCTURE"
                               "Choose the correct spelling:\n"
                               "1. ______\n"
                               "2. ______\n"
                               "3. ______\n"
                               "4. ______\n"
                               "-> save the generated string and append it as the first element of the list 'tasks'\n"
                               "-> provide the correct answer (number of the correct option) and append it as the first element of of the list 'solution'\n"
                               "3. generate fill-in-the-gap task that would help to learn the definition of this lexical item; use the following structure:\n"
                               "STRUCTURE"
                               "Fill in the gap: 'This is a sample _____':\n"
                               "1. ______\n"
                               "2. ______\n"
                               "3. ______\n"
                               "4. ______\n"
                               "-> save the generated string and append it as the second element of the list 'tasks'\n"
                               "-> provide the correct answer (number of the correct option) and append it as the second element of of the list 'solution'\n"
                               "4. generate multiple choice that would help to learn the translation of this lexical item into Russian; use the following structure:\n"
                               "STRUCTURE"
                               "Translate the word ____ into Russian':\n"
                               "1. ______\n"
                               "2. ______\n"
                               "3. ______\n"
                               "4. ______\n"
                               "-> save the generated string and append it as the third element of the list 'tasks'\n"
                               "-> provide the correct answer (number of the correct option) and append it as the third element of of the list 'solution'\n"

                },
                {
                    "role": "user",
                    "content": f"THE INPUT: {item['WORD']}"
                }
            ]
        )
        return ast.literal_eval(output.choices[0].message.content)

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

    base_dictionary = {
        'type': [],
        'item': [],
        'cefr': [],
        'task': []
    }

    for d in data_[:2]:
        temp = get_words(d)
        print(temp)
        colocations = [classify_words(items, wordlist) for items in temp['COLLOCATION'][:2] if len(items) > 0]
        idioms = [classify_words(items, wordlist) for items in temp['IDIOM'][:2] if len(items) > 0]
        verbs = [classify_words(items, wordlist) for items in temp['VERB'][:2] if len(items) > 0]
        words = [classify_words(items, wordlist) for items in temp['WORD'][:2] if len(items) > 0]

        for item in colocations:
            print(item)
            tasks = gen_tasks(item)
            print(tasks)
            if tasks != 'NaN':
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
            print(item)
            tasks = gen_tasks(item)
            print(tasks)
            if tasks != 'NaN':
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
            print(item)
            tasks = gen_tasks(item)
            print(tasks)
            if tasks != 'NaN':
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
            print(item)
            tasks = gen_tasks(item)
            print(tasks)
            if tasks != 'NaN':
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
