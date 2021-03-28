from models_handler import predict
from models_handler import get_models


def main():
    models = get_models()
    resnet18 = models['resnet18']

    res = predict(resnet18, '../dataset/imagenet-mini/val/n01806143/ILSVRC2012_val_00022422.JPEG')

    print(res)


if __name__ == '__main__':
    main()
