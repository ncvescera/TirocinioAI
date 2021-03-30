import json
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image


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
    # img_size: sarà la nuova dimenzione dell'immagine.
    #           Serve per alcuni modelli come la InceptionV3 che necessitano di immagini superiori ad una certa dimenzione
    def prepare_image(self, img_path: str, img_size=(224, 244)) -> torch.tensor:
        img = Image.open(img_path).convert('RGB')       # forza la lettura dell'immagine come RGB
        img = transforms.ToTensor()(img).unsqueeze_(0)  # converte l'immagine in un tensore per essere normalizzata
                                                        # serve per rendere più facile la normalizzazione e la classificazione
        
        # altro modo per la normalizzazione che e' un po bruttino
        # normalize = transforms.Normalize(
        #    mean=[0.485, 0.456, 0.406],
        #    std=[0.229, 0.224, 0.225]
        #)

        # istruzioni per la preparazione dell'immagine
        normalize = transforms.Compose([
                transforms.Resize(img_size),    # scala l'immagine con le date misure
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # transforms.ToTensor()
        ])

        img = normalize(img)  # normalizza l'immagine per poter essere classificata

        return img

    # classifica l'immagine dato un qualunque modello
    # Ritorna una lista contenente 5 dizioniri (TOP 5) del tipo:
    #   {'class': 'classe predetta', 'probability': 0.23341231}
    #
    # model: modello per classificare l'immagine
    # image: tensore pronto per essere classificato
    def predict_proto(self, model, image: torch.tensor) -> list:
        model.eval()    # deve essere messo durante se non si fa training
        
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
