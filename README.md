# industry-classifier

This is a deep learning based visual inspection system for industrial quality control.  
The system takes photo of product, and outputs whether the product is defective.  
  
Feature extraction is based on VGG-16 (with batch normalization), while FCN (Fully Convolutional Network) is used for classifier.  
Transfer learning is supported to work on smaller datasets.  
  
All scripts are implemented with PyTorch and tested on Windows 10 and Ubuntu 16.04 LTS.  

## Dependencies
```
conda install pytorch cuda80 -c pytorch
pip install -r requirements.txt
```

## Datasets
A simple PyTorch dataset implementation is provided.  
  
For training, train and val datasets are needed.  
After each epoch of training, weights will be saved to checkpoint if val gets better loss than before.  
For testing, only test dataset is needed.  
  
Directory structure:
```
root
   |__train
   |      |__class 1
   |      |__class 2
   |      ...
   |      |__class n
   |__val
   |    |__class 1
   |    |__class 2
   |    ...
   |    |__class n
   |__test
         |__class 1
         |__class 2
         ...
         |__class n
```

## Training
```
usage: train.py [-h] [--lr LR] [--end_epoch END_EPOCH] [--transfer]
                [--lock_feature] [--resume] [--checkpoint CHECKPOINT]
                [--root ROOT] [--device DEVICE]

PyTorch VGG Classifier Training

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               learning rate
  --end_epoch END_EPOCH
                        epcoh to stop training
  --transfer            use vgg pretrained feature layers for transfer
                        learning
  --lock_feature        lock vgg featrue layers
  --resume, -r          resume from checkpoint
  --checkpoint CHECKPOINT
                        checkpoint file path
  --root ROOT           dataset root path
  --device DEVICE       device (cuda / cpu)
```
The training script supports transfer learning.
You could lock feature layer parameters to train a baseline network.  
If --transfer is specified, use --checkpoint to load local pre-trained weights, and --resume is not needed.  

## Testing
```
usage: test.py [-h] [--checkpoint CHECKPOINT] [--root ROOT]

PyTorch VGG Classifier Testing

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
                        checkpoint file path
  --root ROOT           dataset root path
```
Just load checkpoint and preform test.  
