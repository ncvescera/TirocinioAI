import csv
import json
import ast
from predict_multi import csv_header
from sys import argv


def get_data(file: str) -> list:
    result = []
    header = csv_header.split(';')

    with open(file, 'r') as csv_file:
        data = csv.reader(csv_file, delimiter=';')

        for elem in data:
            tmp_dict = {}

            for i in range(len(header)):
                tmp_dict[header[i]] = elem[i]

            result.append(tmp_dict)

    # la prima riga e' l'intestazione
    return result[1:]


def main(args):
    for file in args:
        data_dict = get_data(file)
        
        top1 = 0
        top5 = 0
        totali = len(data_dict)

        for elem in data_dict:
            guess = elem['guess_class'].split(' ')[0]
            real = elem['real_class']

            # top 1
            if guess == real:
                top1 += 1

            # top 5
            others_str = elem['other_predictions']
            others_arr = ast.literal_eval(others_str) # da stringa normale (no json o altre formattazioni) trasforma in array/dizionario

            for item in others_arr:
                if real == item['class'].split(' ')[0]:
                    top5 += 1

        print(f' --- {file.upper()} ---')
        print("Guess T1: ", top1)
        print("Guess T5: ", top5)
        print("Total: ", totali)
        print("TOP 1: ", top1/totali)
        print("TOP 5: ", top5/totali)
        print()

if __name__ == "__main__":

    main(argv[1:])