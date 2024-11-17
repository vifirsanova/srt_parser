#!/usr/bin/env python3
from huggingface_hub import InferenceClient

import argparse

import re, json

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

# Download necessary NLTK resource
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

def llm_streaming(filepath):
    with open(filepath, "r") as file:
        data = file.read()

    model_name = "Qwen/Qwen2.5-72B-Instruct"
    client = InferenceClient(model_name, token='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')
    output = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "you are english teacher assistant\n"
                                          "you help to build dictionary for B1-B2 CEFR\n"
                                          "return list of idioms, phrasal verbs and collocations\n"
                                          "do not provide explanations\n"
                                          "focus on short and widespread phrases\n"
                                          "format the output as python dictionary with the following structure\n"
                                          "{idioms: [list of idioms], phrasal verbs: [list of phrasal verbs], collocations: [list of collocations]}"},
            {"role": "user",
             "content": f"return list of idioms, phrasal verbs and collocations for the following data:\n\n {data}"},
        ],
        stream=True,
        max_tokens=2048,
    )

    llm_response = ''.join([chunk.choices[0].delta.content for chunk in output])
    pattern = re.compile(r'\{.*?\}', re.DOTALL)
    # Find all matches
    matches = pattern.findall(llm_response)

    # Print the matches
    if matches:
        dictionary_data = json.loads(matches[0])
        return dictionary_data
    else:
        return {"output": llm_response}

def extract_unique_lemmas_and_collocations(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]

    # Extract bigram collocations based on frequency
    bigram_finder = BigramCollocationFinder.from_words(tokens)
    bigram_finder.apply_freq_filter(1)  # Filter out bigrams that occur less than 1 time
    collocations = [' '.join(bigram) for bigram in bigram_finder.nbest(BigramAssocMeasures.raw_freq, 10)]

    # Combine lemmas and collocations
    all_terms = set(lemmas + collocations)

    # Return the sorted list of unique terms
    return sorted(all_terms)

def clean_srt(srt_data):
    cleaned_lines = []
    lines = srt_data.split('\n')

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

    return '\n'.join(cleaned_lines)

def main(filepath):
    with open(filepath, 'r', encoding='latin1') as f:
        data = f.read()

    cleaned_srt = clean_srt(data)
    pattern = re.compile(r"[^\w\s']")
    cleaned_srt = re.sub(pattern, ' ', cleaned_srt)
    cleaned_srt = [x.replace('\n', ' ').lower() for x in cleaned_srt.split('\n\n')]

    unique_terms = extract_unique_lemmas_and_collocations(' '.join(cleaned_srt))

    with open(f"{filepath}_cleaned.txt", "w") as file:
        for item in cleaned_srt:
            file.write(f"{item}\n")

    with open(f"{filepath}_lemmas.txt", "w") as file:
        for item in unique_terms:
            file.write(f"{item}\n")

    with open(f"{filepath}_phrases.json", "w") as json_file:
        json.dump(llm_streaming(f"{filepath}_cleaned.txt"), json_file, indent=4)

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="*.srt parser for NLP, takes filepath to *.srt as input")

    # Add arguments
    parser.add_argument("filepath", type=str, help="A path to *.srt file")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.filepath)