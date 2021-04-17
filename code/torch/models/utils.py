from os import listdir


csv_header = 'image;probability;guess_class;real_class;other_predictions'


class PredictionData:
    """Rappresenta i dati di una classificazione
        Serve per salvare in modo semplice i dati in csv
    """

    def __init__(self, model:str, image:str, real_class:str, data:list):
        self.model = model
        self.image = image
        self.data = data
        self.real_class = real_class
    
    def to_csv(self):
        return f'{self.image};{self.data[0]["probability"]};{self.data[0]["class"]};{self.real_class};{self.data}'


def get_all_dirs(path: str) -> list:
    """Prende tutte le cartelle nel path passatogli

        Parameters:
            path (str): path dove andare a prendere le cartelle

        Return:
            list[str]: tutte le cartelle presenti in quel path
    """

    res = listdir(path)

    # crea una variabile globale e la setta al path passatogli
    # serve per far funzionare la funzione get_all_dirs_files()
    # [va rivisto]
    global dataset_dir
    dataset_dir = path

    return res


def get_all_dirs_files(folder: str) -> list:
    """Prende tutti i file nella cartella passatagli

        Parameters:
            folder (str): cartella nella quale sono contenuti i file

        Return:
            list[str]: tutti i file presenti in quella cartella
    """

    files = listdir(f'{dataset_dir}/{folder}')

    # crea la lista di files
    res = []
    for elem in files:
        res.append(f'{dataset_dir}/{folder}/{elem}')

    return res


def save_csv(name: str, predictions: list):
    """Salva tutte le predizioni di un modello in un file csv

        Parameters:
            name (str)                          : nome del modello (sarà anche il nome del file)
            prediction (list[PredictionData])   : lista contenente tutte le predizioni del modello (PredictData)
    """

    f = open(f'{name}.csv', 'w')
    f.write(csv_header)
    f.write('\n') # i ritorni a capo sono importanti sennò si rompe il file csv

    for pred in predictions:
        f.write(pred.to_csv())
        f.write('\n')

    f.close()