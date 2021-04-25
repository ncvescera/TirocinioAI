from os import listdir


dataset_dir = ''    # variabile globale per ricordarsi il path del dataset
csv_header = 'image;probability;guess_class;real_class;other_predictions'


def to_dict(arr: list):
    """Trasforma il risultato della classificazione in un array
        di dizionari pronto per essere utilizzato dagli altri script

        Parameters:
            array (list): array che deve essere trasformato
                            array[0]: class_number  (n01223984)
                            array[1]: class_name    (sea_snake)
                            array[2]: confidenza    (0.234123)
    """

    result = []

    for elem in arr:
        tmp_dict = {}

        tmp_dict['probability'] = elem[2]
        tmp_dict['class'] = f'{elem[0]} {elem[1]}'

        result.append(tmp_dict)

    return result


def get_all_dirs(path: str) -> list:
    """Prende tutte le cartelle nel path passatogli

        Parameters:
            path (str): il path della cartella da cui si vuole prendere il contenuto

        Return:
            list[str]: tutte le cartelle/file contenute nel path
    """

    res = listdir(path)

    global dataset_dir
    dataset_dir = path

    return res


def get_all_dirs_files(folder: str) -> list:
    """Prende tutti i files nella cartella passatagli.
        Gli aggiunge anche il path relativo.

        Parameters:
            folder (str): path della cartella da cui prendere i file.

        Return:
            list[str]: file e cartelle con il path relativo contenute nel path dato.
                        (se nella cartella ci sono altre cartelle, queste saranno incluse)
                        (non fa distinzione cartelle e file)
    """

    files = listdir(f'{dataset_dir}/{folder}')

    res = []
    for elem in files:
        res.append(f'{dataset_dir}/{folder}/{elem}')

    return res


class PredictionData:
    """Classe che contiene tutti i dati delle predizioni di un modello
        utilizzata per salvare i dati in csv
    """

    def __init__(self, model, image, real_class, data):
        self.model = model
        self.image = image
        self.data = data
        self.real_class = real_class
    
    def to_csv(self):
        return f'{self.image};{self.data[0]["probability"]};{self.data[0]["class"]};{self.real_class};{self.data}'


def save_csv(name: str, predictions: list):
    """Salva i dati di un modello in csv

        Parameters:
            name (str)                          : nome del modello (sarà il nome che avrà il file csv)
            predictions (list[PredictionData])  : lista con tutte le predizioni del modello (PredictionData)
    """

    # scrive i risultati su file csv
    f = open(f'{name}.csv', 'w')
    f.write(csv_header)
    f.write('\n') # i ritorni a capo sono importanti per il csv

    for pred in predictions:
        f.write(pred.to_csv())
        f.write('\n')

    f.close()