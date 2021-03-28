from os import listdir


csv_header = 'image;probability;guess_class;real_class;other_predictions'

class PredictionData:
    def __init__(self, model, image, real_class, data):
        self.model = model
        self.image = image
        self.data = data
        self.real_class = real_class
    
    def to_csv(self):
        return f'{self.image};{self.data[0]["probability"]};{self.data[0]["class"]};{self.real_class};{self.data}'



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


def save_csv(name: str, predictions: list):
    # scrive i risultati su file csv
    f = open(f'{name}.csv', 'w')
    f.write(csv_header)
    f.write('\n')

    for pred in predictions:
        f.write(pred.to_csv())
        f.write('\n')

    f.close()