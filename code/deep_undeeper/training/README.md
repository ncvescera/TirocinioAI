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

## Modello

```
Model: "DeepUndeeper"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0
_________________________________________________________________
conv2d (Conv2D)              (None, 224, 224, 512)     14336
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 112, 112, 512)     0
_________________________________________________________________
batch_normalization (BatchNo (None, 112, 112, 512)     2048
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 112, 112, 256)     1179904
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 56, 56, 256)       0
_________________________________________________________________
batch_normalization_1 (Batch (None, 56, 56, 256)       1024
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 56, 56, 64)        147520
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 28, 28, 64)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 28, 28, 512)       295424
_________________________________________________________________
up_sampling2d (UpSampling2D) (None, 56, 56, 512)       0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 56, 56, 256)       1179904
_________________________________________________________________
up_sampling2d_1 (UpSampling2 (None, 112, 112, 256)     0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 112, 112, 128)     295040
_________________________________________________________________
up_sampling2d_2 (UpSampling2 (None, 224, 224, 128)     0
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 224, 224, 3)       3459
=================================================================
Total params: 3,118,659
Trainable params: 3,117,123
Non-trainable params: 1,536
_________________________________________________________________
None
```
