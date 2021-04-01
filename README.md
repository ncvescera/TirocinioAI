# TirocinioAI
Dati e Codice prodotti durante il tirocinio

## Jupyter Notebooks

Se non hai il coraggio di affrontare un'installazione in locale sono disponibili i notebook per effettuare la fase di testing su Google Colab:

* [Keras_Model_Testing.ipynb](./code/keras/Keras_Model_Testing.ipynb)
* [PyTorch_Model_Testing.ipynb](./code/torch/PyTorch_Model_Testing.ipynb)

Se sei un vero uomo 🐒 segui le istruzioni successive !

## Installazione

### Preparazione

Prima di installare `Keras` e `PyTorch` è necessario installare `virtualenv` per evitare che i pacchetti e le loro dipendenze possano entrare in conflitto.

Installare i seguenti pacchetti:

```
sudo apt update
sudo apt install python3-dev python3-pip python3-venv
```

Creare due cartelle distinte per le 2 librerie oppure utilizzare quelle già presenti nella cartella: `code/keras` e `code/torch`.<br>
Ricordare che se si creano 2 cartelle nuove ci va poi copiato il codice.

### Keras

Spostarsi sulla cartella destinata a `Keras` ed eseguire i seguenti comandi:

```
python3 -m venv --system-site-packages ./venv
source ./venv/bin/acrivate

pip install --upgrade pip
pip install --upgrade tensorflow
pip install keras
deactivate
```

### PyTorch

Spostarsi sulla cartella destinata a `PyTorch` ed eseguire i seguenti comandi:

```
python3 -m venv --system-site-packages ./venv
source ./venv/bin/acrivate

pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
deactivate
```

Installo solo la versione CPU in quanto per il testing non è necessario l'utilizzo della GPU.

## Dataset

Il dataset utilizzato per il testing è il seguente:

[ImageNet Mini](https://www.kaggle.com/ifigotin/imagenetmini-1000)

Va posizionato nella stessa directory delle cartelle adibite al codice di `Keras` e `Pytorch`:

```
----- ./
---------- keras
---------- torch
---------- dataset
```

## Utilizzo

La cartella dove è installato il progetto deve essere del tipo:

```
----- ./
---------- keras
---------- torch
---------- dataset
```

Prima di far partire gli script assicurarsi di aver modificato correttamente la variabile `dataset_dir` (che si trova all'interno dei file `main.py`) con il corretto percorso della cartela del dataset (l'impostazione di default dovrebbe andare bene ma non si sa mai 🚀)

Spostarsi nella cartella dei modelli da testare (e.g `keras`) ed eseguire i seguenti comandi:

```
source ./venv/bin/activate
python main.py
deactivate
```

La prima riga attiva l'ambiente virtuale, la seconda avvia lo script e l'ultima serve per uscire dall'ambiente virtuale.

La stessa cosa vale per `PyTorch`

