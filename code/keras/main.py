import models
from threading import Thread
import argparse


dataset_dir = ''
MAX_THREAD_NUMBER = 3       # numero massi di thread che possono essere eseguiti contemporaneamente


def menu(all=False):
    """Chiede all'utente di scegliere quali modelli testare (tra quelli disponibili).

        Parameters:
            all=False (bool): se 'True' ritorna tutti i modelli senza stampare nulla a video.

        Return:
            list[ProtoModel]: lista contenente i modelli scelti dall'utente da testare
    """

    # prende tutti i modelli disponibili
    modelli = models.get_models()
    modelli = list(modelli.values())

    if all:
        return modelli

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


def download_all():
    """Scarica tutti i modelli disponibili
        Andrebbe eseguita prima della fase di testing per assicurarsi che siano presenti tutti i modelli
    """

    modelli = models.get_models()

    for key, modello in modelli.items():
        modello


def from_file(file_name: str, modelli, gscale):
    """Testa i modelli scelti con tutti i dataset presenti nel file.
        Per ogni dataset crea una cartella dove sposta tutti i dati (file .csv creati dalla fase di testing)
        ottenuti dal testing.
        Il file deve contenere una riga per path, tipo:
            
            ./path/to/dataset1
            ../path/to/dataset2
            /home/path/to/dataset3/
        
        Parameters:
            file_name (str)             : nome del file che contiene tutti i path dei dataset da utilizzare
            modelli (list[ProtoModel])  : lista con tutti i modelli da testare
            gscale (bool)               : se 'True' applica il filtro GrayScale alle immagini
    """

    # legge tutti i dataset dal file e li mette in una lista
    datasets = []
    with open(file_name, 'r') as f:
        for line in f:
            to_append = line.strip('\n')
            to_append = to_append[:-1] if to_append[-1] == '/' else to_append
            
            datasets.append(to_append)

    # TODO: questa parte va migliorata per renderla indipendente dal OS ed evitare l'uso specifico di comandi shell
    import os
    global dataset_dir
    
    for dataset in datasets:
        dataset_dir = dataset

        # avvia il testing multithreading o single threading
        threading_handler(modelli, gscale)

        new_dir = dataset_dir.split('/')[-1]    # prende il nome da dare alla nuova cartella

        # TODO: evitare comandi shell in questo modo !!
        os.system(f'mkdir {new_dir}')            # crea la cartella del relativo dataset
        os.system(f'mv *.csv {new_dir}')        # sposta tutti i csv dentro la cartella


def threading_handler(modelli, gscale, noThread=False):
    """Funzione che gestisce il multithreading

        Parameters:
            modelli (list[ProtoModel])  : lista con tutti i modelli da testare
            gscale (bool)               : se 'True' applica il filtro GrayScale alle immagini
            noThread=False (bool)       : se 'True' disabilita il MultiThreading
    """

    def worker(model, gscale: bool):
        """Testa un dato Modello.

            Parameters:
                model (ProtoModel)  : modello da testare
                gscale (bool)       : se 'True' applica il filtro GrayScale alle immagini
        """

        model.test(dataset_dir, gscale)

    # esegue il testing in singleThread
    if noThread:
        for model in modelli:
                worker(model, gscale)
        
        return
    
    ts = []     # lista contenente i thread attivi
    i = 0       # variabile per contare quanti thread sono stati creati
    for model in modelli:
        t = Thread(target=worker, args=(model, gscale,))    # creazione del thread
        ts.append(t)                                        # aggiunta del thread alla lista
        t.start()                                           # il thread viene avviato
        
        i += 1

        # quando 3 thread vengono avviati aspetta la loro fine
        # per poi avvirane altri 3 al ciclo successivo
        if i >= MAX_THREAD_NUMBER:
            for t in ts:
                t.join()

            ts = []             # resetta la lista
            i = 0               # resetta il contatore

    # attende la fine degli ultimi thread avviati
    # non è sempre detto che i thread finiscano nel for precedente
    for t in ts:
        t.join()


def main(args):
    # setta il path del dataset
    global dataset_dir
    dataset_dir = args.input[:-1] if args.input[-1] == '/' else args.input

    gscale = args.grayscale

    if args.download:
        print('Download all Models data ...')
        download_all()
        return

    modelli = menu(all=args.all)

    if args.file:
        from_file(dataset_dir, modelli, gscale)
    else: 
        threading_handler(modelli, gscale, noThread=args.nothreads)


if __name__ == "__main__":
    # creazione del parser
    parser = argparse.ArgumentParser(description="Script per testare alcuni modelli di Image Classification di Keras")

    # definizione degli argomenti che accetta lo script
    parser.add_argument("-g", "--grayscale", help="Applica a tutte le immaigni il filtro GrayScale.", action="store_true")
    parser.add_argument("-d", "--download", help="Scarica i modelli e termina lo script.", action="store_true")
    parser.add_argument("--nothreads", help="Esegue il testing in meniera sequenziale senza multithreading", action="store_true")
    parser.add_argument("-i", "--input", type=str, help="Path del dataset (o del file) con cui testare i modelli", required=True)
    parser.add_argument("-a", "--all", help="Testa tutti i modelli disponibili senza stampare il menu di scelta iniziale", action="store_true")
    parser.add_argument("-f", "--file", help="Indica che il path dell'argomento input e' un file e testera' i modelli con ogni path presente nel file.", action="store_true")

    # crea gli argomenti da passare alla funzione main
    args = parser.parse_args()

    main(args)
