# SRCNN - Super-Resolution Convolutional Neural Network
This is the implementation of SRCNN as described in ["Image Super-Resolution Using Deep Convolutional Networks" - Dong et al.](https://arxiv.org/pdf/1501.00092.pdf) for image super-resolution.

SRCNN is one of the first works that uses deep neural network to perform image super-resolution. SRCNN is a pre-upsampling model, which means it first upsamples the low-resolution input image (for example, using bicubic interpolation) before forwarding it through the network to obtain high-resolution result. SRCNN's network architecture is very small with only around 8k parameters, yet it still outperforms bicubic upsampling.

## Data preparation
The network is trained on the *T91* dataset and validated on the *Set5* dataset. These datasets can be downloaded from [here](http://vllab.ucmerced.edu/wlai24/LapSRN). This is the project page of LapSRN, another super-resolution DNN. After the datasets have been downloaded and extracted, please go to *config.py* to change the directory of the training and validation set.

## Training
I found the network weight initialization scheme and optimizing using SGD as described in the paper is slow and hard to optimize, although it leads to better result. Also, as I only want to quickly try out SRCNN, I use the default weight initialization of PyTorch and Adam optimizer for training. The learning rate starts at *1e-3* and divides by *2* everytime the training loss plateaus. Training stops when learning rate is less than *1e-5* or validation PSNR does not improve after 200 epochs.

Compared with the authors' SRCNN trained on T91, my results are better.  Training SRCNN on Imagenet will be left as future work.

## Evaluation
Performance of the network is evaluated using the conventional benchmark of this literature - PSNR metric. To ensure that we get the correct PSNR number, it is advised [here](https://github.com/twtygqyy/pytorch-LapSRN) that we should use the MATLAB function (psnr, rgb2ycbcr, ycbcr2rgb, etc.) for evaluating. However, as this project is only for learning purpose and switching between Python and MATLAB is troublesome, I try to re-implement these functions (they are put in *utilities.py*).

As instructed by the [NTIRE2017 challenge](http://www.vision.ee.ethz.ch/~timofter/publications/Timofte-CVPRW-2017.pdf), a rim of *s+2*, where *s* is the upscaling factor, is ignored during computing PSNR.

## Usage
### Training
```
usage: train.py [-h] --upscale_factor UPSCALE_FACTOR [--batch_size BATCH_SIZE]
                [--use_new_lr USE_NEW_LR] [--n_epochs N_EPOCHS]
                [--checkpoints_dir CHECKPOINTS_DIR]
                [--reload_from RELOAD_FROM] [--start_epoch START_EPOCH]
                [--seed SEED] [--cuda]

optional arguments:
  -h, --help            show this help message and exit
  --upscale_factor      super resolution upscale factor
  --batch_size          batch size for training
  --use_new_lr          lr to use, otherwise use from config.py
  --n_epochs            number of epochs to train
  --checkpoints_dir     directory to save checkpoints
  --reload_from         directory to checkpoint to resume training from
  --start_epoch         epoch number to start from
  --seed SEED           random seed
  --cuda                whether to use cuda
```
An example of training:
```
$ python train.py --upscale_factor 2 --batch_size 64 --n_epochs 1000 --checkpoints_dir checkpoints --cuda
```
Train a SRCNN network with upscaling factor 2, batchsize 64 for 1000 epochs using CUDA. Checkpoint after each epoch is saved to *checkpoints* folder. 

### Evaluate on an image set
```
usage: evaluate.py [-h] --image_dir IMAGE_DIR --model MODEL --upscale_factor
                   UPSCALE_FACTOR [--cuda]

optional arguments:
  -h, --help            show this help message and exit
  --image_dir           directory to image set for evaluating PSNR
  --model MODEL         model file
  --upscale_factor      upscale factor
  --cuda                whether to use cuda
```
An example of evaluating:
```
$ python evaluate.py --image_dir ../../datasets/super-resolution/Set5 --upscale_factor 3 --model trained_model/SRCNN_upscale_3.pth
```
Evaluate the trained model SRCNN_upscale_3.pth on the Set5 dataset with upscaling factor 3.

### Super-resolve an image
```
usage: super_resolve.py [-h] --model MODEL --upscale_factor UPSCALE_FACTOR
                        --input INPUT --output OUTPUT

optional arguments:
  -h, --help            show this help message and exit
  --model               model file
  --upscale_factor      upscale factor
  --input INPUT         input image to super resolve
  --output OUTPUT       where to save the output image
```
An example to super-resolve an image:
```
$ python super_resolve.py --model trained_model/SRCNN_upscale_2.pth --upscale_factor 2 --input inp.png --output out.png
```

## Experimental results

| DataSet | x2 upscaling (PSNR) | x3 upscaling (PSNR) | x4 upscaling (PSNR) |
| ------- |:-------------------:|:-------------------:|:--------------------:
| Set5    | 36.59               | 32.53               | 30.17               |
| Set14   | 32.46               | 29.16               | 27.34               |

The trained models that achieve these results are put in folder *trained_model*.

## References
* ["Image Super-Resolution Using Deep Convolutional Networks" - Dong et al.](https://arxiv.org/pdf/1501.00092.pdf)
* [PyTorch example of super-resolution DNN](https://github.com/pytorch/examples/tree/master/super_resolution)
* [Dataset fron LapSRN project](http://vllab.ucmerced.edu/wlai24/LapSRN)