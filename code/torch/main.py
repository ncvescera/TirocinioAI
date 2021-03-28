from sys import argv
from threading import Thread

from models_handler import predict, get_models
import utils


dataset_dir = '../dataset/imagenet-mini/val'


def menu():
    models = get_models()

    print('Questi sono i modelli disponibili:')
    print()

    keys = list(models.keys())

    i = 1
    for key in keys:
        print(f'\t{i}) {key}')
        i += 1
    print()

    print("Scegli quali modelli vuoi testare: ['a' Tutti, '1' Solo quello scelto, '1 4 2 3' Solo quelli scelti]")

    scelta = input()
    scelta = scelta.split(' ')

    result = {}
    if scelta[0] == 'a':
        result = models
    else:
        for elem in scelta:
            # result.append(models[keys[int(elem)-1]])
            result[keys[int(elem)-1]] = models[keys[int(elem)-1]]

    return result

        
def worker(model, name, all_classes):
    predictions = []

    for clas in all_classes:
        print(f'**** {name.upper()}  {clas.upper()} ****')

        for image in utils.get_all_dirs_files(clas):
            res = predict(model, image)

            pred = utils.PredictionData(name, image, clas, res)
            predictions.append(pred)

    utils.save_csv(name, predictions)


def main():
    models = menu()
    
    all_dirs = utils.get_all_dirs(dataset_dir)

    ts = []
    for name, model in models.items():
        t = Thread(target=worker, args=(model, name, all_dirs,))
        ts.append(t)
        t.start()

    for t in ts:
        t.join()


if __name__ == '__main__':
    main()
