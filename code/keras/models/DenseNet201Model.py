import tensorflow
import numpy as np
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input, decode_predictions
from .utils import to_dict
from .utils import get_all_dirs
from .utils import get_all_dirs_files
from .utils import PredictionData
from .utils import save_csv


# Classe che implementa le funzioni del modello DenseNet201
class DenseNet201Model:
    def __init__(self):
        self.model = DenseNet201(weights='imagenet')
        self.name = 'DenseNet201'
    
    # classifica una lista di immagini date
    # Ritorna una lista contenente dizionari del tipo:
    #   {'class': 'classe predetta', 'probability': 0.2348172}
    # Ogni elemento della lista è una predizione: contiene i primi 5 risultati (TOP 5)
    # Ci saranno tanti elementi nella lista finali quante immagini
    #
    # gscale: se 'True' le immagini verranno convertite in GrayScale
    def predict(self, imgs: list, gscale=False) -> list:
        results = []
        for img in imgs:
            img_path = img

            # sceglie se caricare l'immagine in grayscale o rgb
            # di defaul è RGB
            img_type = 'rgb'
            if gscale:
                img_type = 'grayscale'

            # carica l'immagine con la grandezza minima richiesta dal modello
            img = image.load_img(img_path, target_size=(224, 224), color_mode=img_type)

            # questo passaggio va fatto perchè l'immagine caricata in GrayScale ha una forma che non va bene
            # al classificatore ! Va quindi aperta in GrayScale e poi convertita in RGB per avere la giusta forma.
            # L'immagine rimane in bianco e nero ma così piace al classificatore
            if gscale:
                img = img.convert('RGB')

            # processa l'immagine per la classificazione
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # classifica l'immagine
            preds = self.model.predict(x)
            
            # trasforma i risultati in un dizionario
            decoded =  decode_predictions(preds, top=5)[0]

            results.append(to_dict(decoded))
        
        return results
    
    # testa il modello con tutte le immagini del dataset passatogli
    def test(self, dataset_path: str, gscale=False):
        all_classes = get_all_dirs(dataset_path)    # prende tutte le cartelle (classi) del dataset
        
        # classifica tutti i file (divisi per classe) del dataset
        predictions = []
        for clas in all_classes:
            print(f'**** {self.name.upper()}  {clas.upper()} ****')

            for image in get_all_dirs_files(clas):
                res = self.predict([image], gscale)[0]  # prende il primo risultato dato che la lista ritornata
                                                        # avrà un unico risultato (gli viene passata solo 1 immagine)
                
                # print('. ', end='')
                
                # salva i dati delle predizioni per essere poi salvati in csv
                pred = PredictionData(self.name, image, clas, res)
                predictions.append(pred)
            
            # print()

        # salva le predizioni in csv
        save_csv(self.name, predictions)
       