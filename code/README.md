# Codice

Nelle cartelle `keras` e `torch` ci sono gli script in python per il testing dei modelli.

Ricordarsi sempre di modificare la variabile `dataset_dir` (presente nei file `main.py`) con il corretto percorso della cartella del dataset.<br>
Nel dubbio controllare sempre quella di default e provarla.

## Main.py

Il file `main.py`, presente sia in `Keras` che in `PyTorch`, serve per avviare la fase di testing dei rispettivi modelli.

Con il comando `python main.py --help` è possibile stampare a video la pagina di aiuto.

```
python main.py --help

usage: main.py [-h] [-g] [-d] [--nothreads]

Script per testare alcuni modelli di Image Classification di Keras

optional arguments:
  -h, --help       show this help message and exit
  -g, --grayscale  Applica a tutte le immaigni il filtro GrayScale.
  -d, --download   Scarica i modelli e termina lo script.
  --nothreads      Esegue il testing in meniera sequenziale senza multithreading
```

Di seguito una breve spiegazione degli argomenti che accetta `main.py`:

* `-g` o `--grayscale`: serve per avviare il test in modalità **GrayScale**. In questa modalità tutte le immagini di qualsiasi modello verranno prima trasformate in bianco e nero (GrayScale) e successivamente passate la classificatore.
* `-d` o `--download`: controlla se tutti i modelli necessari al funzionamento dello script sono stati già scaricati, in caso contrario li scarica ed interrompe lo script.
* `--nothreads`: forza il funzionamento dello script in modalità **single-thread**. Lo script di defaulto funziona in multithreading: per ogni modello scelto avvia un nuovo thread che avrà il compito di gestire il test del singolo modello. Vengono avviati **3 thread** alla volta per evitare che si occupi troppa CPU. Il numero di thread massimi alla volta può essere sempre modificato cambiando il valore della variabile `MAX_THREAD_NUMBER` presente all'inizio del file `main.py`

## Analyze.py

Lo script `analyze.py` serve per estrapolare le statistiche dai vari file csv generati durante la fase di testing.<br>
Si usa nel seguente modo:

```
python analyze.py file_csv1.csv file_csv2.csv ... file_csvn.csv
```

L'output sarà una cosa del tipo:

```
 --- FILE_CSV1.CSV ---
Guess T1:  2920
Guess T5:  3620
Total:  3923
TOP 1:  0.7443283201631404
TOP 5:  0.9227631914351262

 --- FILE_CSV2.CSV ---
Guess T1:  2920
Guess T5:  3620
Total:  3923
TOP 1:  0.7443283201631404
TOP 5:  0.9227631914351262

...

```

Dove:

* la prima riga indica il nome del file
* `Guess T1` indica quante immagini ha classificato correttamente (solo il primo risultato)
* `Guess T5` indica quante immagini sono state classificate correttamente (rientrano tra i primi 5 risultati)
* `Total` il numero totale di immagini passategli come testing
* `TOP1` la precisione (`accuracy`) TOP 1 (solo i primi risultati è quello corretto)
* `TOP 5` la precisione ('accuracy') TOP 5 (uno tra i primi 5 è il risultato corretto)

## Aggiungere Modelli

Se si vogliono aggiungere modelli basta controllare uno dei file presenti nella cartella `models` (sia di `keras` che di `torch`) e creare un nuovo file di conseguenza.

Per ogni nuovo file aggiunto va modificato il file `__init.py__`: nella funzione `get_models()` aggiungere una nuova entry al dizionario (c'è solo quello non puoi sbagliarti).
