from sys import argv
import requests
import json
from os import listdir
from threading import Thread

url = 'http://localhost/predictions'
ping = 'http://localhost/ping'
dataset_dir = 'dataset/imagenet-mini/val'
ais = ['alexnet', 'squeeze', 'mobile', 'inception', 'resnet']
csv_header = 'image;probability;guess_class;real_class;other_predictions'


class PredictionData:
    def __init__(self, model, image, real_class, data):
        self.model = model
        self.image = image
        self.data = data
        self.real_class = real_class
    
    def to_csv(self):
        return f'{self.image};{self.data[0]["probability"]};{self.data[0]["class"]};{self.real_class};{self.data}'


def predict(model: str, img_path: str) -> list:
    # prende l'immagine
    try:
        image = open(img_path, 'rb').read()
    except Exception as e:
        print('Problemi ad aprire l\'immagine :/')
        print()
        print(e)
        return None

    # manda la richiesta e prende il risultato
    res = requests.post(f'{url}/{model}', data=image)
    
    # stampa il risultato
    # print(img_path)
    # print(res.text)

    res_array = json.loads(res.text)
    # print(res_array[0])
    
    return res_array


# prende tutte le cartelle nel path passatogli
def get_all_dirs(path: str) -> list:
    res = listdir(path)

    return res

# prende tutti i files nella cartella passatagli
def get_all_dirs_files(folder: str) -> list:
    files = listdir(f'{dataset_dir}/{folder}')

    res = []
    for elem in files:
        res.append(f'{dataset_dir}/{folder}/{elem}')

    return res


# funzione worker per il multithreading
def worker(model, all_classes):
    predictions = []
    # classifica tutti i file (divisi per classe) del dataset
    for clas in all_classes:
        print(f'**** {model.upper()}  {clas.upper()} ****')

        for image in get_all_dirs_files(clas):
            res = predict(model, image)
            
            # print('. ', end='')

            pred = PredictionData(model, image, clas, res)
            predictions.append(pred)
        
        print()

    # scrive i risultati su file csv
    f = open(f'{model}.csv', 'w')
    f.write(csv_header)
    f.write('\n')
    
    for pred in predictions:
        f.write(pred.to_csv())
        f.write('\n')

    f.close()


def main(args):
    # prende tutte le classi nel dataset
    all_classes = get_all_dirs(dataset_dir)

    '''
    # testa ogni modello
    for ai in ais:
        predictions = []
        # classifica tutti i file (divisi per classe) del dataset
        for clas in all_classes:
            print(f'**** {clas.upper()} ****')

            for image in get_all_dirs_files(clas):
                res = predict(ai, image)
                
                print('. ', end='')

                pred = PredictionData(ai, image, clas, res)
                predictions.append(pred)
            
            print()

        # scrive i risultati su file csv
        f = open(f'{ai}.csv', 'w')
        f.write(csv_header)

        for pred in predictions:
            f.write(pred.to_csv())
            f.write('\n')

        f.close()
    '''
    
    ts = []
    for ai in ais:
        t = Thread(target=worker, args=(ai, all_classes,))
        ts.append(t)
        t.start()

    for t in ts:
        t.join()
    

if __name__ == '__main__':
    args = argv[1:]
    main(args)
