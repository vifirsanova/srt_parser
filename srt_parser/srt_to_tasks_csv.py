#!/usr/bin/env python3
import argparse, json, re
import pandas as pd
from huggingface_hub import InferenceClient

def llm_streaming(data, dictionary):
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
             "content": f"Analyze the following list and return dictionary with task descriptions and solutions: \n\n {data}"},
        ],
        stream=True,
        max_tokens=2048,
        temperature=0.7,
    )

    llm_response = ''.join([chunk.choices[0].delta.content for chunk in output])
    pattern = re.compile(r'\[.*?\]', re.DOTALL)
    # Find all matches
    matches = pattern.findall(llm_response)

    for elem in matches[0].split('\n'):
        elem = elem.strip()
        elem = elem.split("'")

        if len(elem) > 2:
            if elem[1] == 'word':
                dictionary['word'].append(elem[-2])
            elif elem[1] == 'solution':
                dictionary['solution'].append(elem[-2])
            elif elem[1] == 'task':
                dictionary['task'].append(elem[-2])

def main(filepath):

    dictionary = {'word': [], 'task': [], 'solution': []}

    with open(f'{filepath}.json', 'r') as f:
        data = json.load(f)

    data_ = []

    for elem in data:
        for item in data[elem]:
            if list(item.values())[0] == 'B1':
                data_.append(list(item.keys())[0])

    n = k = len(data_) // 50
    j = 0

    for i in range(k):
        llm_streaming(data_[j:k], dictionary)
        j += n
        k += n
        print(dictionary)
        print(k)

    df = pd.DataFrame(dictionary)
    df.to_csv('result.csv', index=False)

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="*.srt parser for NLP, takes filepath to *.srt as input")

    # Add arguments
    parser.add_argument("filepath", type=str, help="A path to *.srt file")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.filepath)