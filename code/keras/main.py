from os import listdir
from sys import argv
import models
from threading import Thread


dataset_dir = '../dataset/imagenet-mini/val'


# funzione utilizzata per il multithreading
# Testa un dato modello
#
# model: modello da testare
def worker(model):
    model.test(dataset_dir)


# permette all'utente di scegliere quali modelli testare (tra quelli disponibili)
def menu():
    # prende tutti i modelli disponibili
    modelli = models.get_models()
    modelli = list(modelli.values())

    print("Questi sono i modelli disponibili: ")
    print()

    i = 1
    for model in modelli:
        print(f'\t{i})  {model.name}')
        i += 1

    print()
    print("Quali vuoi scegliere ? ['a' Tutti, '1' Solo quello scelto, '1 3 2 4' Solo quelli elencati]")

    scelta = input()
    scelta = scelta.split(' ')

    result = []
    if scelta[0] == 'a':
        result = modelli
    else:
        result = []
        for elem in scelta:
            result.append(modelli[int(elem)-1])

    return result


# scarica tutti i modelli disponibili
# andrebbe eseguita prima della fase di testing per assicurarsi 
# che siano presenti tutti i modelli
def download_all():
    modelli = models.get_models()

    for key, modello in modelli.items():
        modello


def main(args):
    # prova a scaricare tutti i modelli
    if len(args) > 0 and args[0] == '-d':
        print('Download all Models data ...')
        download_all()
        return
    
    modelli = menu()

    ts = []
    for model in modelli:
        t = Thread(target=worker, args=(model,))
        ts.append(t)
        t.start()

    for t in ts:
        t.join()
    
    # testing del modello InceptionV3
    # modelli['InceptionV3'].test(dataset_dir)


if __name__ == "__main__":
    main(argv[1:])
