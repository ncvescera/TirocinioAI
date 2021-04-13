import json
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from .utils import get_all_dirs, get_all_dirs_files, PredictionData, save_csv


# Classe che deve essere estesa dalle varie classi che rappresentano i modelli
# Contiene i metodi per caricare le varie classi di ImageNet, preparare l'immagine per essere passata al modello
# e per interpretare i risultati della classificazione
class ProtoModel:
    def __init__(self):
        self.classes_file_path = './imagenet_class_index.json'
        self.classes = self.load_classes()  # carica le classi di ImageNet

    # ritorna una lista che rappresenta le classi di ImageNet
    # E' formata come segue:
    #       0 -> prima classe di ImageNet
    #       1 -> seconda classe di ImageNet
    #       ...
    def load_classes(self) -> list:
        class_idx = json.load(open(self.classes_file_path, 'r')) # converte il file json in array
        idx2label = [f'{class_idx[str(k)][0]} {class_idx[str(k)][1]}' for k in range(len(class_idx))] # crea la lista

        return idx2label

    # apre l'immagine passata (img_path) e la prepara per la classificazione
    #
    # img_path: percorso dove si trova l'immagine
    # img_resize, img_crop: sono valori specifici per ogni modello 
    #                       e servono per adattare l'immagine all'input del classificatore
    def prepare_image(self, img_path: str, img_resize: int, img_crop: int) -> torch.tensor:
        img = Image.open(img_path).convert('RGB')   # apre l'immagine e forza la lettura in RGB
                                                    # serve per impedire l'apertura di alcune immagini in GrayScale

        normalize = transforms.Compose([
                transforms.Resize(img_resize),
                transforms.CenterCrop(img_crop),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = normalize(img)
        input_batch = input_tensor.unsqueeze(0) # adatta la forma dell'immagine modificata per il modello

        return input_batch


    # classifica l'immagine dato un qualunque modello
    # Ritorna una lista contenente 5 dizioniri (TOP 5) del tipo:
    #   {'class': 'classe predetta', 'probability': 0.23341231}
    #
    # model: modello per classificare l'immagine
    # image: tensore pronto per essere classificato
    def predict_proto(self, model, image: torch.tensor) -> list:
        out = model(image) # classifica l'immagine

        # prende i primi 5 risultati con relative precisioni
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        
        _, indices = torch.sort(out, descending=True)
        result_tmp = [(self.classes[idx], percentage[idx].item()) for idx in indices[0][:5]]

        # crea l'array di dizionari di output
        result = []
        for label, prob in result_tmp:
            result.append({'class': label, 'probability': prob/100})

        return result

    # testa il modello con tutte le immagini del dataset passatogli.
    # scrive un file CSV con i risultati del test.
    #
    # name: nome del modello che sara' il nome del file csv e verra' utilizzato per alcune stampe
    # dataset_path: percorso del dataset utilizzato per il test
    # predict_function: funzione utilizzata per la predizione della singola immagine
    def proto_test(self, name: str, dataset_path: str, predict_function):
        all_classes = get_all_dirs(dataset_path)    # prende tutte le cartelle (classi) del dataset
        predictions = []                            # lista per salvare tutte le predizioni

        for clas in all_classes:
            print(f'**** {name.upper()}  {clas.upper()} ****')

            # classifica ogni immagine della cartella (classe) e salva i risultati
            for image in get_all_dirs_files(clas):
                res = predict_function(image)

                pred = PredictionData(name, image, clas, res)
                predictions.append(pred)

        # salva i risultati in un file csv
        save_csv(name, predictions)
