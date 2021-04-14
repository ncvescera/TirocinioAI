from os import listdir, mkdir
from shutil import copyfile

dataset_path = './originali'

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
    
    dest_foledr = './ALine'     # nome della cartella che contiene il nuovo dataset
    
    # crea la nuova cartella col dataset
    # se esiste gia' continua 
    try:
        mkdir(dest_foledr)
    except OSError as error:
        print(f'La cartella esiste gia ({error})')
    
    for img in imgs:
        # crea la cartella della relativa classe
        cls_folder = f'{dest_foledr}/{img["class"]}'
        try:
            mkdir(cls_folder)
        except OSError as error:
            print(f'La cartella esiste gia ({error})')

        # copia la foto dalla vecchia cartella alla nuova
        src = f'{dataset_path}/{img["img"]}'
        dst = f'{cls_folder}/{img["img"]}'
        
        copyfile(src, dst)

if __name__ == '__main__':
    
    imgs = imgs_and_numbers()
    imgs = number_to_badclass(imgs)

    assoc = bad_to_godd_assoc()
    
    res = adapt_dataset(imgs, assoc)

    make_dataset(res)
    
   