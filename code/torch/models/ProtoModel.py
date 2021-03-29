import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import json


class ProtoModel:
    def __init__(self):
        self.classes_file_path = '/mnt/A08E3ED08E3E9EAA/AI/torch/imagenet_class_index.json'
        self.classes = self.load_classes()  # carica le classi di ImageNet

    def load_classes(self):
        class_idx = json.load(open(self.classes_file_path, 'r'))
        idx2label = [f'{class_idx[str(k)][0]} {class_idx[str(k)][1]}' for k in range(len(class_idx))]

        return idx2label

    def prepare_image(self, img_path, img_size=(224, 244)):
        img = Image.open(img_path).convert('RGB')   # forza la lettura dell'immagine come RGB
        img = transforms.ToTensor()(img).unsqueeze_(0)  # converte l'immagine in un tensore per essere normalizzata
        
        # altro modo per la normalizzazione che e' un po bruttino
        # normalize = transforms.Normalize(
        #    mean=[0.485, 0.456, 0.406],
        #    std=[0.229, 0.224, 0.225]
        #)

        normalize = transforms.Compose([
                transforms.Resize(img_size),    # scala l'immagine con le date misure
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # transforms.ToTensor()
        ])

        img = normalize(img)  # normalizza l'immagine per poter essere classificata

        return img

    def predict_proto(self, model, image):
        model.eval()    # deve essere messo durante se non si fa training
        
        # classifica l'immagine
        out = model(image)

        # prende i primi 5 risultati con relative precisioni
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        _, indices = torch.sort(out, descending=True)
        result_tmp = [(self.classes[idx], percentage[idx].item()) for idx in indices[0][:5]]

        # crea l'array di dizionari di output
        result = []
        for label, prob in result_tmp:
            result.append({'class': label, 'probability': prob/100})

        return result
