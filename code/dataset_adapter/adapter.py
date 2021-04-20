from os import listdir, mkdir
from shutil import copyfile
from distutils.dir_util import copy_tree
import argparse
import diego


dataset_path = ''    # path della cartella contenente il dataset da adattare
dest_folder = ''     # nome della cartella che contiene il nuovo dataset
seeds = [            # seeds per rendere ripetibili i test
    69,     # il numero che ti resta nella testa
    666,    # the number of the beast
    777,    # the number of F.C.
    911,    # Nine Eleven Tower Divers
    119,    # Revers 911
    420,    # WeedTime
    1308,   # Anno Domini Unipg
    7,      # il numero massimo di ogni cosa
    23,     # il numero bello :3
    9       # La fine (kill -9 1)
]


def imgs_and_numbers():
    """Associa ad ogni immagine il suo numero progressivo

        Return:
            numbered_imgs: lista di dizionari che rappresentano le immagini
    """

    immagini = listdir(dataset_path)
    numbered_imgs = []

    for img in immagini:
        img_number = int(img.split('_')[-1][:-5])

        numbered_imgs.append({'img': img, 'number': img_number})

    return numbered_imgs


def number_to_badclass(imgs):
    """Associa l'immagine (identificata con un numero progressivo) alla relativa classe in formato numerico (brutto)

        Parameters:
            imgs (list[dict{img, number}]): lista di dizionari che rappresentano le immagini

        Return:
            imgs: lista di dizionari che rappresentano le immagini con in piu' il campo 'class'
    """

    with open('ILSVRC2014_clsloc_validation_ground_truth.txt', 'r') as f:
        index_to_badclass = []
        for line in f:
            fixed_line = line.strip('\n')
            index_to_badclass.append(fixed_line)

    for img in imgs:
        img['class'] = index_to_badclass[img['number']-1]

    return imgs


def bad_to_godd_assoc():
    """Associa ogni numero di classe alla relativa forma ILSVRC

        Return:
            dict: associazione tra numero di classe e forma ILSVRC
    """

    class_assoc = {}
    with open('map_clsloc.txt', 'r') as f:
        for line in f:
            fixed_line = line.strip('\n')
            tmp = fixed_line.split(' ')
            
            bad = tmp[1]    # numero di classe
            good = tmp[0]   # forma in ILSVRC

            class_assoc[bad] = good

    return class_assoc


def adapt_dataset(imgs, assoc):
    """Aggiunge alla singola immagine la sua giusta classe (ILSVRC) facendo il mapping tra ILSVRC e l'altro formato

        Parameters:
            imgs (list[dict{img, number, class}]): lista di dizionari che rappresentano le immagini
            assoc (dict): associa ogni numero di classe alla rispettiva forma ILSVRC

        Return:
            imgs (list[dict{img, number, class}]): lista di dizionari che rappresentano le immagini
    """

    for img in imgs:
        img['class'] = assoc[str(img['class'])]

    return imgs


def make_dataset(imgs):
    """Crea il dataset spostando ogni immagine in una cartella con il nome della classe che rappresenta

        Parameters:
            imgs (list[dict{img, number, class}]): lista di dizionari che rappresentano le immagini 
    """
   
    # crea la nuova cartella col dataset
    # se esiste gia' continua 
    try:
        mkdir(dest_folder)
    except OSError as error:
        print(f'La cartella esiste gia ({error})')
    
    for img in imgs:
        # crea la cartella della relativa classe
        cls_folder = f'{dest_folder}/{img["class"]}'
        try:
            mkdir(cls_folder)
        except OSError as error:
            print(f'La cartella esiste gia ({error})')

        # copia la foto dalla vecchia cartella alla nuova
        src = f'{dataset_path}/{img["img"]}'
        dst = f'{cls_folder}/{img["img"]}'
        
        copyfile(src, dst)


def start_adapting():
    """Avvia la procedura di adattamento del dataset
    """

    imgs = imgs_and_numbers()
    imgs = number_to_badclass(imgs)

    assoc = bad_to_godd_assoc()
    
    res = adapt_dataset(imgs, assoc)

    make_dataset(res)


def main(args):
    # controllo presenza di Input e Output
    if args.input is None or args.output is None:
        print('Input o Output non settati.')
        return

    # modifica delle 2 variabili globali
    global dataset_path
    dataset_path = args.input[:-1] if args.input[-1] == '/' else args.input     # elimina l'ultimo / se presente nel path

    global dest_folder
    dest_folder = args.output[:-1] if args.output[-1] == '/' else args.output   # elimina l'ultimo / se presente nel path

    if args.diego:
        # creazione cartella Images per input di Diego
        try:
            mkdir('./Images')
        except OSError as error:
            print(f'La cartella esiste gia ({error})')
        
        # creazione cartella Filtered per output di Diego
        try:
            mkdir('./Filtered')
        except OSError as error:
            print(f'La cartella esiste gia ({error})')

        # copia del dataset da adattare nella cartella Images per essere filtrate
        copy_tree(dataset_path, './Images')
        
        # modifica dei path di input
        dataset_path = './Filtered'
        initial_dest_folder = dest_folder   # utilizzata per dare il nome alla cartella finale

        for seed in seeds:
            print(f"**** SEED {seed} ****")
            # modifica dei path di input in funzione del seed
            dest_folder = f'{initial_dest_folder}_{seed}'

            diego.apply_random_filters(seed)
            start_adapting()

            print()

    else:
        start_adapting()


if __name__ == '__main__':
    # creazione del parser
    parser = argparse.ArgumentParser(description="Script per testare alcuni modelli di Image Classification di Keras")

    # definizione degli argomenti che accetta lo script
    parser.add_argument("-i", "--input", help="Path della cartella dove e' contenuto il dataset da adattare", type=str)
    parser.add_argument("-o", "--output", help="Path della cartella dove verra' creato il nuovo dataset", type=str)
    parser.add_argument("-d", "--diego", help="Applica dei filtri alle immagini prima di creare il nuovo dataset", action="store_true")

    # crea gli argomenti da passare alla funzione main
    args = parser.parse_args()

    main(args)
    
    
   