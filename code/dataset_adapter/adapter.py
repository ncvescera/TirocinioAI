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


def gingham():
    """Applica il filtro GinGham alle immagini della cartella data.
        python -i ../destinazione/dataset -o ../cartella/dataset -g
        
        '../destinazione/dataset' indica la cartella che contiene tutte le immagini
        '../cartella/dataset' indica dove verra' salvato il nuovo dataset.
            Dentro a questa cartella verra' creata una nuova cartella con il nome 'Gam_dataset'
            (dataset e' il nome della cartella in input)

        python -i ./in.txt -o ../cartella/dataset -g

        './in.txt' e' un file con dentro path di dataset. Deve avere per forza estensione .txt !!
            Fa la stessa cosa come sopra per tutti i dataset indicati in quel file
    """

    import gimgum
    from os import walk, path

    def gingham_procedure(in_path: str):
        global dataset_path
        global dest_folder

        dest_path = './Filtered'

        # prende tutte le immagini a cui applicare i filtri
        contenuto = walk(in_path)  # prende il contenuto di tutta la cartella e delle cartelle al suo interno

        imgs = []   # tutte le immagini a cui applicare il filtro
        for root, dirs, files in contenuto:
            for name in files:
                tmp_name :str = path.join(root, name)                           # prende il path del singolo file all'interno delle varie cartele
                if tmp_name.endswith('.JPEG') or tmp_name.endswith('.PNG'):     # se e' un immagini lo aggiunge
                    imgs.append(tmp_name)

        # applica il filtro alle immagini e le salva
        for img in imgs:
            gimgum.apply_gingham(img, dest_path)

        in_name = in_path.split('/')[-1] # prende il nome della cartella in input

        # imposto le variabili di input e otput
        dataset_path = './Filtered'
        dest_folder += f'/Gam_{in_name}'

        start_adapting()    # avvia l'adapting del dataset

    # creazione cartella Filtered per output gingham
    try:
        mkdir('./Filtered')
    except OSError as error:
        print(f'La cartella esiste gia ({error})')

    if dataset_path.endswith('.txt'):
        # prende tutti i dataset da file
        datasets = []
        with open(dataset_path, 'r') as f:
            for line in f:
                datasets.append(line.strip('\n'))
        
        # per ogni dataset avvia la procedura gingham
        global dest_folder
        reset_dest_folder = dest_folder
        for dataset in datasets:
            gingham_procedure(dataset)
            dest_folder = reset_dest_folder # reset della destinazione per evitare che le cartelle siano nestate

    else:
        gingham_procedure(dataset_path)


def start_adapting():
    """Avvia la procedura di adattamento del dataset
    """

    imgs = imgs_and_numbers()
    imgs = number_to_badclass(imgs)

    assoc = bad_to_godd_assoc()
    
    res = adapt_dataset(imgs, assoc)

    make_dataset(res)


def main(args):
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

    elif args.gingham:
        gingham()

    else:
        start_adapting()


if __name__ == '__main__':
    # creazione del parser
    parser = argparse.ArgumentParser(description="Script per testare alcuni modelli di Image Classification di Keras")

    # definizione degli argomenti che accetta lo script
    parser.add_argument("-i", "--input", help="Path della cartella dove e' contenuto il dataset da adattare", type=str, required=True)
    parser.add_argument("-o", "--output", help="Path della cartella dove verra' creato il nuovo dataset", type=str, required=True)
    parser.add_argument("-d", "--diego", help="Applica dei filtri alle immagini prima di creare il nuovo dataset", action="store_true")
    parser.add_argument(
        "-g", 
        "--gingham", 
        help="Applica il filtro GinGham alle immagini contenute nella cartella di input. Come output va indicata la cartella dove andra' creata la nuova cartella.\
            Puo' essere passato come input un file .txt (deve avere questa estensione per forza !!) contenente tutti i path dei dataset a cui applicare il filtro", 
        action="store_true"
        )

    # crea gli argomenti da passare alla funzione main
    args = parser.parse_args()

    main(args)
    
    
   