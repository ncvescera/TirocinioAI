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
    
    def proto_predict(self, imgs: list, tsize: tuple, model, preprocess_input, decode_predictions, gscale=False) -> list:
        """Classifica una lista di immagini date

            Parameters:
                tsize (tuple(int, int))         : è una Tupla e rappresenta la dimensione con cui deve essere ridimenzionata l'immagine
                model (PortoModel)              : il modello con cui classificare le immagini
                preprocess_input (Callable)     : funzione per il preprocessing dell'immagine da passare al modello
                decode_predictions (Callable)   : funzione per interpretare il risultato della classificazione
                gscale (bool)                   : se 'True' le immagini verranno convertite in GrayScale

            Return:
                list[dict{'class': str, 'probability': double}]: 
                    Ritorna una lista contenente dizionari del tipo:
                        {'class': 'classe predetta', 'probability': 0.2348172}
                    Ogni elemento della lista è una predizione: contiene i primi 5 risultati (TOP 5)
                    Ci saranno tanti elementi nella lista finali quante immagini
        """

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

    def proto_test(self, dataset_path: str, name: str, predict_function, gscale=False):
        """Testa il modello con tutte le immagini del dataset passatogli e genera un file contenente tutti i risultati delle classificazioni.

            Parameters:
                dataset_path (str)          : path della cartella contenente il dataset
                name (str)                  : nome del modello (servirà come nome del file finale)
                predict_function (Callable) : funzione utilizzata per la classificazione delle immagini
                gscale (bool)               : indica se va applicato il filtro GrayScale
        """
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