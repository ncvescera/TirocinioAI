from os import listdir
from sys import argv
import models
from threading import Thread


dataset_dir = '../dataset/imagenet-mini/val'
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


# stampa a video tutti i possibili argomenti che accetta lo script
# [dovrà essere modificato con argparse]
def help():
    print('Questi sono gli argomenti passabili allo scritp:')
    print()
    print('\t-g\tApplica il filtro GrayScale alle immagini prima di essere classificate')
    print('\t-d\tProva a scaricare tutti i modelli e si ferma')
    print()


def main(args):
    gscale = False

    # prova a scaricare tutti i modelli
    if len(args) > 0:
        if args[0] == '-d':
            print('Download all Models data ...')
            download_all()
            return
        
        if args[0] == '--help':
            help()
            return

        if args[0] == '-g':
            gscale = True
    
    modelli = menu()

    for model in modelli:
        worker(model, gscale)

    '''
    # vecchia versione multithrading
    # permette di eseguire troppi thread contemporaneamente
    # mettendo in crisi il sistema
    ts = []
    for model in modelli:
        t = Thread(target=worker, args=(model,))
        ts.append(t)
        t.start()

    for t in ts:
        t.join()
    '''

    '''
    # probabile versione multithreading sensata
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
    '''
    # testing del modello InceptionV3
    # modelli['InceptionV3'].test(dataset_dir)


if __name__ == "__main__":
    main(argv[1:])
