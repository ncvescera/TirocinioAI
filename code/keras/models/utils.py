from os import listdir


# trasforma il risultato della classificazione in un array
# di dizionari pronto per essere utilizzato dagli altri script
def to_dict(arr: list):
    result = []

    for elem in arr:
        tmp_dict = {}

        tmp_dict['probability'] = elem[2]
        tmp_dict['class'] = f'{elem[0]} {elem[1]}'

        result.append(tmp_dict)

    return result


dataset_dir = ''

# prende tutte le cartelle nel path passatogli
def get_all_dirs(path: str) -> list:
    res = listdir(path)

    global dataset_dir
    dataset_dir = path

    return res

# prende tutti i files nella cartella passatagli
def get_all_dirs_files(folder: str) -> list:
    files = listdir(f'{dataset_dir}/{folder}')

    res = []
    for elem in files:
        res.append(f'{dataset_dir}/{folder}/{elem}')

    return res


csv_header = 'image;probability;guess_class;real_class;other_predictions'


# classe che contiene tutti i dati delle predizioni di un modello
# utilizzata per salvare i dati in csv
class PredictionData:
    def __init__(self, model, image, real_class, data):
        self.model = model
        self.image = image
        self.data = data
        self.real_class = real_class
    
    def to_csv(self):
        return f'{self.image};{self.data[0]["probability"]};{self.data[0]["class"]};{self.real_class};{self.data}'


# salva i dati di un modello in csv
#
# name: nome del modello (sarà il nome che avrà il file csv)
# predictions: lista con tutte le predizioni del modello (PredictionData)
def save_csv(name: str, predictions: list):
    # scrive i risultati su file csv
    f = open(f'{name}.csv', 'w')
    f.write(csv_header)
    f.write('\n') # i ritorni a capo sono importanti per il csv

    for pred in predictions:
        f.write(pred.to_csv())
        f.write('\n')

    f.close()