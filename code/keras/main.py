import models
from threading import Thread
import argparse


dataset_dir = '../dataset/imagenet-mini/val'
aline = '../dataset/ALine'
MAX_THREAD_NUMBER = 3       # numero massi di thread che possono essere eseguiti contemporaneamente


# funzione utilizzata per il multithreading
# Testa un dato modello
#
# model: modello da testare
def worker(model, gscale: bool):
    model.test(dataset_dir, gscale)


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
    # controlla se il dataset scelto e' giusto
    if args.dataset is not None:
        if args.dataset == 'ALine':
            global dataset_dir
            dataset_dir = aline
        
        elif args.dataset == 'ImageNet':
            pass
        
        else:   # se la scelta e' sbagliata interrompe il programma
            print('Dataset inesistente, controlla meglio !')
            return

    gscale = args.grayscale

    if args.download:
        print('Download all Models data ...')
        download_all()
        return

    modelli = menu()

    # avvia la modalità sequenziale
    if args.nothreads:
        for model in modelli:
            worker(model, gscale)
    
        return
    
    # versione multithreading sensata
    ts = None   # lista contenente i thread attivi
    i = 0       # variabile per contare quanti thread sono stati creati
    for model in modelli:
        ts = []                                             # inizializza la nuova lista

        t = Thread(target=worker, args=(model, gscale,))    # creazione del thread
        ts.append(t)                                        # aggiunta del thread alla lista
        t.start()                                           # il thread viene avviato
        
        i += 1

        # quando 3 thread vengono avviati aspetta la loro fine
        # per poi avvirane altri 3 al ciclo successivo
        if i == MAX_THREAD_NUMBER:
            for t in ts:
                t.join()
            
            i = 0

    # attende la fine degli ultimi thread avviati
    # non è sempre detto che i thread finiscano nel for precedente
    for t in ts:
        t.join()


if __name__ == "__main__":
    # creazione del parser
    parser = argparse.ArgumentParser(description="Script per testare alcuni modelli di Image Classification di Keras")

    # definizione degli argomenti che accetta lo script
    parser.add_argument("-g", "--grayscale", help="Applica a tutte le immaigni il filtro GrayScale.", action="store_true")
    parser.add_argument("-d", "--download", help="Scarica i modelli e termina lo script.", action="store_true")
    parser.add_argument("--nothreads", help="Esegue il testing in meniera sequenziale senza multithreading", action="store_true")
    parser.add_argument("--dataset", type=str, help="Seleziona il dataset con cui testare i modelli.\nScelte: ALine | ImageNet")

    # crea gli argomenti da passare alla funzione main
    args = parser.parse_args()

    main(args)
