import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import json


def _load_classes(classes_file_path):
    class_idx = json.load(open(classes_file_path, 'r'))
    idx2label = [f'{class_idx[str(k)][0]} {class_idx[str(k)][1]}' for k in range(len(class_idx))]

    return idx2label


def _prepare_image(img_path):
    img = Image.open(img_path)
    img = transforms.ToTensor()(img).unsqueeze_(0)  # converte l'immagine in un tensore per essere normalizzata
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    
    img = normalize(img)  # normalizza l'immagine per poter essere classificata

    return img


def _predict(model, image, classes):
    model.eval()
    
    # classifica l'immagine
    out = model(image)

    # prende i primi 5 risultati con relative precisioni
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    result_tmp = [(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]

    # crea l'array di dizionari di output
    result = []
    for label, prob in result_tmp:
        result.append({'class': label, 'probability': prob/100})

    return result


def predict(model, image):
    classes = _load_classes('imagenet_class_index.json')
    img = _prepare_image(image)
    res = _predict(model, img, classes)

    return res


def get_models():
    modelli = {
        'resnet18': models.resnet18(pretrained=True)
    }

    return modelli