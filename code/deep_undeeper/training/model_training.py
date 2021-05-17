import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Input, Conv2D, UpSampling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator


EPOCHS  = 200   # indica il numero di epoche da utilizzare nella fase di training
BATCH   = 8     # indica il numero di batch da utilizzare nella fase di training


def create_model() -> Model:
    """ Prepara il modello che analizza le immagini
        
        Return:
            Model: modello della rete
    """

    x = Input(shape=(224, 224, 3))      # reso input shape standard

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
    d_conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(up2) # ho dovuto aggiungere padding=same
    up3 = UpSampling2D((2, 2))(d_conv3)
    r = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up3)

    model = Model(x, r)
    model.compile(optimizer='adam', loss='mse')

    return model


def prepare_images(dataset_path: str, seed=None, shuffle=True) -> DirectoryIterator:
    """ Prepara le immagini da passare alla rete.
        Tutte le immagini vengono normalizzate (pixel * (1./255)) e ridimenzionate a 224x224.

        Parameters:
            dataset_path (str): il path della cartella dove sonon contenute le immagini.
                                La cartella deve contenere altre cartelle che rappresentano la classe delle immagini contenute.

        Return:
            DirectoryIterator: iteratore da passare alla rete con tutte le immagini preparate e pronte all'utilizzo.
    """
    # definizione di ImageDataGenerator per caricare in modo efficiente le immagini
    datagen :ImageDataGenerator = ImageDataGenerator(
            data_format = "channels_last",  # serve per avere un array del tipo (sample, height, width, channels)
            rescale=1./255                  # normalizzazione (??) delle immagini
                                            # fa la stessa cosa che con CV2 [(pixel-255)/255]        
        )

    # carica le immagini dalla cartella del dataset
    result :DirectoryIterator = datagen.flow_from_directory(
            dataset_path,                       # path della cartella del dataset
            target_size=(224, 224),             # (height, width) resize che verrà fatto all'immagine
            batch_size=BATCH,                   # 16 occupano troppa memoria per la 1080
            class_mode='input',                 # indica il tipo dell'arry delle classi. 'input' ritorna immagini uguali a quelle di input     
            seed=seed,                          # imposta il seed per rendere i test ripetibili. (di default e' None)
            shuffle=shuffle                     # ogni volta cambia l'ordine delle immagini (le mischia)
        )

    return result


def main(args):
    # setto le variabili globali per la fase di training
    global EPOCHS
    global BATCH

    BATCH   = args.batch
    EPOCHS  = args.epochs

    # prepara i dataset di training e validation
    training :DirectoryIterator = prepare_images('./ilsvrc2012Training')
    validation :DirectoryIterator = prepare_images('./ilsvrc2012Validation')

    # prende la dimensione dei dataset di training e validation
    training_size   = training.samples        # .samples prende il numero di immagini contenute nell'iteratore
    validation_size = validation.samples

    # creo il modello
    deepundeeper :Model = create_model()
    print(deepundeeper.summary())

    # alleno il modello    
    model_filepath = './deepundeeper_checkpoint.h5' # path dove andranno salvati i checkpoint
    
    # callbacks
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)                                    # Early stopping (stops training when validation doesn't improve for {patience} epochs)
    save_best = ModelCheckpoint(model_filepath, monitor='val_loss', save_best_only=True, mode='min', verbose = 1) # Saves the best version of the model to disk

    deepundeeper_history = deepundeeper.fit(
        training, 
        steps_per_epoch=training_size // BATCH,     # training
        epochs=EPOCHS, 
        validation_data=validation,
        validation_steps=validation_size // BATCH,
        callbacks=[es, save_best]
        )
    
    # deepundeeper_history = deepundeeper.fit(training, epochs=EPOCHS)
    
    # salvo il modello
    deepundeeper.save('deep_undeeper224.model')

    # salvo il grafico
    plt.plot(deepundeeper_history.epoch, deepundeeper_history.history['loss'])
    plt.title('Epochs on Training Loss')
    plt.xlabel('# of Epochs')
    plt.ylabel('Mean Squared Error')
    plt.savefig('model_training_chart.png')


if __name__ == '__main__':
     # creazione del parser
    parser = argparse.ArgumentParser(description="Script per allenare la rete DeepUndeeper224")

    # dichiaro gli argomenti opzionali
    parser.add_argument(
        "-b", 
        "--batch", 
        type=int, 
        help="Numero di batch da utilizzare durante la fase di training", default=8
        )

    parser.add_argument(
        "-e", 
        "--epochs", 
        type=int, 
        help="Numero di epoche da utilizzare durante la fase di training", default=200
        )

    # crea gli argomenti da passare alla funzione main
    args = parser.parse_args()
    
    main(args)