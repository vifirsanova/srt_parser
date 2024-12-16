#!/usr/bin/env python3
import argparse, re, ast, json

def extract_dict_from_string(text, dictionary):
    try:
        # Use ast.literal_eval to safely evaluate the string as a Python literal
        result = ast.literal_eval(text)
        if isinstance(result, dict):
            if 'IDIOMS' in result.keys():
                dictionary['IDIOMS'] += result['IDIOMS']
            if 'VERBS' in result.keys():
                dictionary['VERBS'] += result['VERBS']
            if 'WORDS' in result.keys():
                dictionary['WORDS'] += result['WORDS']
            if 'COLLOCATIONS' in result.keys():
                dictionary['COLLOCATIONS'] += result['COLLOCATIONS']
            #print(dictionary)
    except (ValueError, SyntaxError) as e:
        pass

def main(filepath):

    with open(f'{filepath}_wordlist_.txt', 'r') as f:
        data = f.read()

    pattern = r'`([^`]+)`'

    matches = re.findall(pattern, data)

    dictionary = {'IDIOMS': [], 'VERBS': [], 'WORDS': [], 'COLLOCATIONS': []}

    for m in matches:
        m = m.replace('python', '').strip()
        if '{' in m:
            extract_dict_from_string(m, dictionary)

    with open(f'{filepath}.json', 'w') as f:
        json.dump(dictionary, f)

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="*.srt parser for NLP, takes filepath to *.srt as input")

    # Add arguments
    parser.add_argument("filepath", type=str, help="A path to *.srt file")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.filepath)