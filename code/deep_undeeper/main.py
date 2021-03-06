import argparse
import glob
import numpy as np
import cv2
import gc
from os import walk, path, environ

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # disabilita i log di tensorflow

# Import di KERAS e roba per Modello
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Conv2D, UpSampling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
## ------------------------------- ##

def create_model():
    """Prepara il modello che analizza le immagini

        Return:
            keras.Model: modello della rete
    """

    x = Input(shape=(300, 300, 3)) 

    # Encoder
    e_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    pool1 = MaxPooling2D((2, 2), padding='same')(e_conv1)
    batchnorm_1 = BatchNormalization()(pool1)
    e_conv2 = Conv2D(256, (3, 3), activation='relu', padding='same')(batchnorm_1)#32
    pool2 = MaxPooling2D((2, 2), padding='same')(e_conv2)
    batchnorm_3 = BatchNormalization()(pool2)
    e_conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(batchnorm_3)#64
    h = MaxPooling2D((2, 2), padding='same')(e_conv4)


    # Decoder
    d_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same')(h)
    up1 = UpSampling2D((2, 2))(d_conv1)
    d_conv2 = Conv2D(256, (3, 3), activation='relu', padding='same')(up1)
    up2 = UpSampling2D((2, 2))(d_conv2)
    d_conv3 = Conv2D(128, (3, 3), activation='relu')(up2)
    up3 = UpSampling2D((2, 2))(d_conv3)
    r = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up3)

    model = Model(x, r)
    model.compile(optimizer='adam', loss='mse')

    return model


def prepare_dataset(imgs: list) -> np.array:  # TODO: nome provvisorio, cambiarlo
    """Prepara tutte le immagini del dataset per essere analizzate
        Normalizza limmagine (??) e la ridimensiona a 300x300 px

        Parameters:
            imgs (list[srt]): contiene le immagini (con path relativo) che devono essere preparate

        Return:
            np.array[np.ndarray] (??): ritorna un numpy.arry contenente tutte le immagini (numpy.ndarray probabilmente)
                                        pronte per essere passate al modello. 
    """

    result = []
    for img in imgs:
        cv_img = cv2.imread(img)
        cv_img = (cv_img-255)/255
        resized = cv2.resize(cv_img, (300, 300)) 
        result.append(np.array(resized))

    return np.array(result) # rende tutto un numpy.array per essere passato al modello


def get_all_image_from_dir(dir_path: str) -> list:
    """Prende tutte le immagini contenute in una cartella e nelle sue sottocartelle.

        Parameters:
            dir_path (str): path della cartella contenente le immagini da analizzare

        Return:
            list[str]: immagini (con path relativo) da analizzare
    """

    contenuto = walk(dir_path)  # prende il contenuto di tutta la cartella e delle cartelle al suo interno

    imgs = []
    for root, dirs, files in contenuto:
        for name in files:
            tmp_name :str = path.join(root, name)                           # prende il path del singolo file all'interno delle varie cartele
            if tmp_name.endswith('.JPEG') or tmp_name.endswith('.PNG'):     # se e' un immagini lo aggiunge
                imgs.append(tmp_name)

    return imgs


def save_to_csv(dataset_path:str, pulite:int, distorte:int, errori:np.array):
    """Salva in csv i dati dell'errore del dataset.

        Parameters:
            dataset_path (str): path del dataset che e' stato testato
            pulite (int): numero di immagini considerate pulite
            distorte (int): numero di immagini considerate distorte
            errori (np.array): array contenente tutti gli errori delle singole immgini
    """

    csv_header = "name;tot_imgs;pulite;distorte;errore_avg"

    tot_imgs = pulite + distorte        # totale immagini analizzate
    name = dataset_path.split('/')[-1]  # prende il nome dell'ultima cartella nel path passato

    with open(f'encod_{name}.csv', 'w') as f:
        f.write(csv_header)
        f.write('\n')
        f.write(f'{name};{tot_imgs};{pulite};{distorte};{np.mean(errori)}\n')


def main(args):
    model_path   :str = 'gaussian_blur_uguali_rivolta2.model'

    # --- Crea il Modello della Rete e lo carica da file --- #
    print('Carico il modello ...')

    gaussian_auto_encoder = create_model()
    gaussian_auto_encoder = keras.models.load_model(model_path)
    
    print('... OK')
    # --- -------------------------------------------- --- #

    for dataset in args.input:
        # prepara il nome del dataset, eventualmente toglie l'ultimo '/'
        dataset_path :str = dataset[:-1] if dataset[-1] == '/' else dataset
        
        print(f'Eseguo il test per:\t{dataset_path}')

        # --- Prepara le immagini per la rete --- #
        print('Preparo le immagini ...')
        
        imgs_nofilter :list = get_all_image_from_dir(dataset_path)       # prende tutti i file all'interno della cartella
        imgs_nofilter_ready :np.array = prepare_dataset(imgs_nofilter)   # scala le immagini alla giusta dimensione   

        print('... OK')
        # --- ------------------------------ --- #

        # -- Utilizza il Modello e analizza le immagini --- #
        print('Utilizzo il modello ...')

        input_imgs :np.array    = imgs_nofilter_ready
        input_tensor :tf.Tensor = tf.convert_to_tensor(input_imgs, dtype=tf.float32)    #  bisogna convertire il numpy.arry in tensore per evitare memory leak !!
        risultato  :np.array    = gaussian_auto_encoder.predict(input_tensor)           # il modello controlla tutte le immagini

        # calcola l'errore (??) per ogni immagine
        conta_pulita = 0        # numero di immagini pulite (considerate appartenenti al dataset (??))
        conta_distorta = 0      # numero di immagini distorete (considerate NON appartenenti al dataset (??))
        treshold = 0.87         # treshold di attivazione per considerare un'immagine pulita (<= treshold) o distorata (> treshold)
        errori = np.array([])   # utilizzo un numpy.arry per effettuare i calcoli piu' facilmente alla fine
        
        for i in range(len(input_imgs)):
            errore = (np.square(input_imgs[i] - risultato[i])).mean()   # calcola l'errore (??)
            errore = errore * 100
            errori = np.append(errori, errore)

            # conta quante sono le immagini considerate 'distorte' e quelle 'pulite'
            if errore > treshold :
                conta_distorta += 1
            else:
                conta_pulita += 1

        print(f"immagini pulite = {conta_pulita} ({conta_pulita/len(input_imgs)*100}%)")
        print(f"immagini distorte =  {conta_distorta} ({conta_distorta/len(input_imgs)*100}%)")
        print(f"media errore dataset = {np.mean(errori)}")

        save_to_csv(dataset_path, conta_pulita, conta_distorta, errori)
        _ = gc.collect()    # dovrebbe impedire di utilizzare troppa memoria durante tutto lo script (??)


if __name__ == '__main__':
    # creazione del parser
    parser = argparse.ArgumentParser(description="Script per testare alcuni modelli di Image Classification di Keras")

    # definizione degli argomenti che accetta lo script
    parser.add_argument(
        "input", 
        type=str, 
        nargs='+',
        default=[],
        help="Path del dataset da controllare.\
                E' possibile passare allo script sia una cartella con tutte le immagini dentro sia una cartella formattata come un dataset per essere testato"
    )
    parser.add_argument("--allInOne", help="Indica che tutte le immagini sono contenute all'interno della cartella passata", action="store_true")

    # crea gli argomenti da passare alla funzione main
    args = parser.parse_args()

    main(args)
