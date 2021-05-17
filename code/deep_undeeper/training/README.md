# DeepUndeeper Model Training

In questa cartella c'e' lo script che ho utilizzato per allenare DeepUndeeper224.
La shape di input e' passata da 300x300 a 224x224 (formato molto piu' comune tra i modelli preallenati di keras e pytorch).

Di default utilizza 

```
epochs 	    = 200
batch_size  = 8
```

Con la 1080 una batch_size di 16 risultava eccessiva in quanto occupava tutta la VRAM !!

Il dataset usato e' un subset del validation dataset di ImageNet, ho preso le immagini dalla 40000 fino alla 50000 (comprese).

Con gli argmoenti `--epochs` e `--batch` e' possibile specificare relativamente il numero di epoche e il numero di batch:

```
# default run
python main.py -e 200 -b 8

# oppure
python main.py --epochs 200 --batch 8
```

## Dataset

Il dataset che e' stato utilizzato per l'allenamento e la valutazione del modello e' ImageNet (ILSVRC2012) [link](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar)
