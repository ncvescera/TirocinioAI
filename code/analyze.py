import csv
import json
import ast
import argparse
from os import listdir, system


csv_header = 'image;probability;guess_class;real_class;other_predictions'
no_save = False # se True, non salva i risultati su file.
                # Il file strongest.txt viene sempre salvato !!


def get_data(file: str) -> list:
    result = []
    header = csv_header.split(';')

    with open(file, 'r') as csv_file:
        data = csv.reader(csv_file, delimiter=';')

        for elem in data:
            tmp_dict = {}

            for i in range(len(header)):
                tmp_dict[header[i]] = elem[i]

            result.append(tmp_dict)

    # la prima riga e' l'intestazione
    return result[1:]


def strong_imgs(dirs: list):
    """Controlla quali immagino sono state correttamente classificate in ambedue i file passati

        Parameters:
            dir_normal (str)        : Path della cartella contenente i dati senza filtri
            dir_multifilter (str)   : Path della cartella contenente i dati con i  filtri
    """

    def strong_imgs_top1(file_normal: str, file_multifilter: str) -> list:
        """Controlla quali immagini sono classificate correttamente i entrambi i file passati (solo TOP1).
            Controlla per ogni immagine se e' stata classificata bene sia nel primo che nel secondo file, se lo e'
            viene considerata 'strong'.
            Le immagini 'strong' vengono salvate su un file in formato CSV con il nome del rispettivo modello.

            Parameters:
                file_normal (str)       : Path del file contenente i dati senza filtri
                file_multifilter (str)  : Path del file contenente i dati con i  filtri

            Return:
                list: Tutte le immagini considerate 'strong'
        """

        normal = get_data(file_normal)              # carica i dati del file senza filtri
        multifilter = get_data(file_multifilter)    # carica i dati del file senza filtri

        strong = []                 # immagini classificate bene
        loop_len = len(normal)      # le due liste sono lunghe uguali, una vale l'altra
        for i in range(loop_len):
            normal_check = normal[i]['guess_class'].split(' ')[0] == normal[i]['real_class']                # controlla se l'immagini e' classificata correttamente
            multifilter_check = multifilter[i]['guess_class'].split(' ')[0] == multifilter[i]['real_class'] # controlla se l'immagini e' classificata correttamente
            
            # se l'immagine e' classificata bene sia su normal che su multifilter la aggiunge ai risultati
            if normal_check and multifilter_check:
                strong.append(normal[i])

        # stampa le immagini forti
        normal_name = file_normal.split('/')[-1]            # nome del file senza filtri
        multifilter_name = file_multifilter.split('/')[-1]  # nome del file con i filtri
        
        print(f"{normal_name} VS {multifilter_name}")
        print(f'Total Images: {len(strong)}/{len(normal)}')
        print()
        
        # salva il nome delle foto strong su un file txt
        only_names = [] # contiene solo i nomi delle foto per essere ritornate

        if not no_save:
            with open(f'{normal_name[:-4]}.txt', 'w') as f:     # con -4 elimino il .csv dal nome e lo sostituisco con .txt
                for elem in strong:
                    f.write(elem["image"])
                    f.write('\n')

                    only_names.append(elem["image"])
        
        return only_names
        '''
        # salva le foto con tutti i dati su file csv
        with open(normal_name, 'w') as f:
            # scrive l'header
            f.write(csv_header)
            f.write('\n')
            
            # scrive tutte le immagini forti
            for elem in strong:
                to_write = ''                   # string temporanea che verra' scritta alla fine nel file
                for _, value in elem.items():
                    to_write += f'{value};'
                
                # elimina il ; finale e mette il carattere di fine riga
                to_write = to_write[:-1]
                to_write += '\n'
                
                f.write(to_write)               # scrive la riga nel file
        '''

    def get_all_files(dir_name: str) -> list:
        """Ritorna tutti i file (con il path relativo) contenuti nella cartella passata

            Parameters:
                dir_name (str): Path della cartella

            Return:
                list[str]: tutti i file contenuti nella cartella
        """

        files_normal = []
        for line in listdir(dir_name):
            files_normal.append(f'{dir_name}/{line}')

        return files_normal

    def find_strongest_imgs(strong_imgs: list):
        """Data la lista con tutte le immagini Forti, trova quelle fortissime.
            Le immagini Fortissime (Strongest) sono quelle che sono state classificate bene da tutte le reti
            e con tutti i filtri.

            Parameters:
                strong_imgs (list[list[str]]): la lista e' formata nel seguente modo:
                                                    per ogni posizione della lista c'e' una lista
                                                    contenente tutte le immagini forti.
        """

        strongest = []
        confronto = strong_imgs.pop()   # prendo il primo set di immagini forti

        for img_confronto in confronto:
            trovato = True
            
            # controlla se l'immagine nel set di confronto e' contenuta in tutti gli altri
            # all prima volta che l'immagine non e' contenuta setta trovato a Folse ed interrompe il ciclo
            for imgs in strong_imgs:
                if img_confronto in imgs:
                    pass
                else:
                    trovato = False
                    break
            
            # se l'immagini e' presente in tutti i set la aggiunge alle fortissime
            if trovato:
                strongest.append(img_confronto)

        # salva su file .txt le foto strongest trovate
        with open('strongest.txt', 'w') as f:
            for line in strongest:
                f.write(line)
                f.write('\n')

    dir_normal = dirs[0]            # la prima cartella e' quella senza filtri
    dirs_multifilter = dirs[1:]     # le altre sono considerate con filtri

    normal_files = get_all_files(dir_normal)            # tutti i file contenenti nella cartella senza filtri
    
    strong_imgs = []    # conterra' tutte le immagini considerate forti
    
    # effettua il test strong per ogni cartella passata
    for elem in dirs_multifilter:
        multifilet_files = get_all_files(elem)          # tutti i file contenenti nella cartella con filtri
        
        # per ogni file presente nelle cartelle effettua il test 'strong'
        for i in range(len(normal_files)):
            result = strong_imgs_top1(normal_files[i], multifilet_files[i])
            strong_imgs.append(result)  # aggiunge l'array qui dentro alla lista strong_imgs

        if not no_save:
            # alla fine di ogni test crea una cartella e ci mette dentro i risultati
            # la cartella ha il nome Strong_NomeCartellaConSeed
            new_dir_name = elem.split('/')[-1]
            new_dir_name = f'Strong_{new_dir_name}'
            
            system(f'mkdir {new_dir_name}')
            system(f'mv *.txt {new_dir_name}')

    find_strongest_imgs(strong_imgs)


def main(args):
    global no_save
    no_save = args.nosave
    files = args.input   # prepara i path dei files

    if args.strong:
        strong_imgs(files)
    
    else:
        for file in files:
            data_dict = get_data(file)
            
            top1 = 0
            top5 = 0
            totali = len(data_dict)

            for elem in data_dict:
                guess = elem['guess_class'].split(' ')[0]
                real = elem['real_class']

                # top 1
                if guess == real:
                    top1 += 1

                # top 5
                others_str = elem['other_predictions']
                others_arr = ast.literal_eval(others_str) # da stringa normale (no json o altre formattazioni) trasforma in array/dizionario

                for item in others_arr:
                    if real == item['class'].split(' ')[0]:
                        top5 += 1

            print(f' --- {file.upper()} ---')
            print("Guess T1: ", top1)
            print("Guess T5: ", top5)
            print("Total: ", totali)
            print("TOP 1: ", top1/totali)
            print("TOP 5: ", top5/totali)
            print()


if __name__ == "__main__":
     # creazione del parser
    parser = argparse.ArgumentParser(description="Script per riealborare i dati ottenuti dal Testing dei Modelli")

    # definizione degli argomenti che accetta lo script
    parser.add_argument("-i", "--input", type=str, help="Path dei file(s) da elaborare", nargs='+', default=[], required=True)
    parser.add_argument("--strong", help="Avvia la procedura per controllare quali immagini sono 'forti'. Devono essere inseriti 2 path di due cartelle contenenti i files. Il primo path deve essere quello NON FILTRATO.", action="store_true")
    parser.add_argument("--nosave", help="Evita di salvare i risultati del test su file. Solo le immagini Fortissime vengono salvate !!", action="store_true")

    # crea gli argomenti da passare alla funzione main
    args = parser.parse_args()

    main(args)
