from sys import argv
from threading import Thread
import utils
from models import get_models

dataset_dir = '../dataset/imagenet-mini/val'


# funzione per far scegliere all'utente quali modelli testare
# [non è attualmente usata e va rivista]
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


# funzione che dovrà essere usata per il mutrithreading
# testa il modello e scrive i risultati in un file csv
#
# model: modello che deve essere testato
# all_classes: lista contenente tutte le cartelle (classi) del dataset
def worker(model, all_classes: list):
    predictions = []    # lista per salvare tutte le predizioni

    for clas in all_classes:
        print(f'**** {model.name.upper()}  {clas.upper()} ****')

        # classifica ogni immagine della cartella (classe) e salva i risultati
        for image in utils.get_all_dirs_files(clas):
            res = model.predict(image)

            pred = utils.PredictionData(model.name, image, clas, res)
            predictions.append(pred)

    utils.save_csv(model.name, predictions)


def main():
    modelli = get_models()                      # prende tutti i modelli disponibili
    all_dirs = utils.get_all_dirs(dataset_dir)  # prende tutte le cartelle (classi) del dataset

    # testa ogni modello
    for name, modello in modelli.items():
        worker(modello, all_dirs)

    '''
    mod = AlexNet()
    res = mod.predict('../dataset/imagenet-mini/val/n01806143/ILSVRC2012_val_00022422.JPEG')
    print(res)
    '''
    '''
    models = menu()
    
    all_dirs = utils.get_all_dirs(dataset_dir)

    ts = []
    for name, model in models.items():
        t = Thread(target=worker, args=(model, name, all_dirs,))
        ts.append(t)
        t.start()

    for t in ts:
        t.join()
    '''


if __name__ == '__main__':
    main()
