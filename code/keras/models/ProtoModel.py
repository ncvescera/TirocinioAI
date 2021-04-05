import numpy as np
from tensorflow.keras.preprocessing import image
from .utils import to_dict
from .utils import get_all_dirs
from .utils import get_all_dirs_files
from .utils import PredictionData
from .utils import save_csv


class ProtoModel:
    def __init__(self):
        pass
    
    # classifica una lista di immagini date
    # Ritorna una lista contenente dizionari del tipo:
    #   {'class': 'classe predetta', 'probability': 0.2348172}
    # Ogni elemento della lista è una predizione: contiene i primi 5 risultati (TOP 5)
    # Ci saranno tanti elementi nella lista finali quante immagini
    #
    # tsize: è una Tupla e rappresenta la dimenzione con cui deve essere ridimenzionata l'immagine
    # model: il modello con cui classificare le immagini
    # preprocess_input: funzione per il preprocessing dell'immagine da passare al modello
    # decode_predictions: funzione per interpretare il risultato della classificazione
    # gscale: se 'True' le immagini verranno convertite in GrayScale
    def proto_predict(self, imgs: list, tsize: tuple, model, preprocess_input, decode_predictions, gscale=False) -> list:
        results = []
        for img in imgs:
            img_path = img

            # sceglie se caricare l'immagine in grayscale o rgb
            # di defaul è RGB
            img_type = 'rgb'
            if gscale:
                img_type = 'grayscale'

            # carica l'immagine con la grandezza minima richiesta dal modello
            img = image.load_img(img_path, target_size=tsize, color_mode=img_type)

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
            preds = model.predict(x)
            
            # trasforma i risultati in un dizionario
            decoded =  decode_predictions(preds, top=5)[0]

            results.append(to_dict(decoded))
        
        return results

    # testa il modello con tutte le immagini del dataset passatogli.
    # alla fine genera un file contenente tutti i risultati delle classificazioni
    #
    # dataset_path: path della cartella contenente il dataset
    # name: nome del modello (servirà come nome del file finale)
    # predict_function: funzione utilizzata per la classificazione delle immagini
    # gscale: indica se va applicato il filtro GrayScale
    def proto_test(self, dataset_path: str, name: str, predict_function, gscale=False):
        all_classes = get_all_dirs(dataset_path)    # prende tutte le cartelle (classi) del dataset

        # classifica tutti i file (divisi per classe) del dataset
        predictions = []
        for clas in all_classes:
            print(f'**** {"[GRAY]" if gscale else ""} {name.upper()}  {clas.upper()} ****')

            for image in get_all_dirs_files(clas):
                res = predict_function([image], gscale)[0]  # prende il primo risultato dato che la lista ritornata
                                                            # avrà un unico risultato (gli viene passata solo 1 immagine)
                
                # print('. ', end='')

                # salva i dati delle predizioni per essere poi salvati in csv
                pred = PredictionData(name, image, clas, res)
                predictions.append(pred)
            
            # print()

        # modifica il nome del file per GrayScale
        if gscale:
            name = f'g_{name}'

        # salva le predizioni in csv
        save_csv(name, predictions)