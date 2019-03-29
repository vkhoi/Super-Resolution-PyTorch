# SRCNN - Super-Resolution Convolutional Neural Network
This is the implementation of SRCNN as described in ["Image Super-Resolution Using Deep Convolutional Networks" - Dong et al.](https://arxiv.org/pdf/1501.00092.pdf) for image super-resolution.

SRCNN is one of the first works that uses deep neural network to perform image super-resolution. SRCNN is a pre-upsampling model, which means it first upsamples the low-resolution input image (for example, using bicubic interpolation) before forwarding it through the network to obtain high-resolution result. SRCNN's network architecture is very small with only around 8k parameters, yet it still outperforms bicubic upsampling.

## Data preparation
The network is trained on the *T91* dataset and validated on the *Set5* dataset. Link to download these datasets can be downloaded from [here](http://vllab.ucmerced.edu/wlai24/LapSRN). This is the project page of LapSRN, another super-resolution DNN. After the datasets have been downloaded and extracted, please go to *data.py* to change the directory of the training and validation set.

## Training
I found the network weight initialization scheme and optimizing using SGD as described in the paper is extremely slow and hard to optimize. Therefore, I use the default weight initialization of PyTorch and Adam optimizer for training, which is much faster and helps me successfully reproduce the paper reported results.

## Evaluation
Performance of the network is evaluated using the conventional benchmark of this literature - PSNR metric. To ensure that we get the PSNR performance, it is advised [here](https://github.com/twtygqyy/pytorch-LapSRN) that we should use the MATLAB function (psnr, rgb2ycbcr, ycbcr2rgb, etc.) for evaluating. However, as this project is only for learning purpose and switching between Python and MATLAB is troublesome, I try to re-implement these functions (they are put in *utilities.py*).

I also follow the paper to use the *Set5* dataset for validation. As instructed by the [NTIRE2017 challenge](http://www.vision.ee.ethz.ch/~timofter/publications/Timofte-CVPRW-2017.pdf), a rim of $s + 2$, where $s$ is the upscaling factor, is ignored during computing PSNR.

## Usage
### Training
```
usage: train.py [-h] --upscale_factor UPSCALE_FACTOR [--batch_size BATCH_SIZE]
                [--n_epochs N_EPOCHS] [--checkpoints_dir CHECKPOINTS_DIR]
                [--reload_from RELOAD_FROM] [--start_epoch START_EPOCH]
                [--seed SEED] [--cuda]

optional arguments:
  -h, --help            show this help message and exit
  --upscale_factor      super resolution upscale factor
  --batch_size          batch size for training
  --n_epochs            number of epochs to train
  --checkpoints_dir     directory to save checkpoints
  --reload_from         directory to checkpoint to resume training from
  --start_epoch         epoch number to start from
  --seed                random seed
  --cuda                whether to use cuda
```
An example of training:
```
$ python train.py --upscale_factor 2 --batch_size 64 --n_epochs 80 --checkpoints_dir checkpoints --cuda
```
Train a SRCNN network with upscaling factor 2, batchsize 6 for 80 epochs using CUDA. Checkpoint after each epoch is saved to *checkpoints* folder.

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
$ python evaluate.py -image_dir Set5 --model checkpoints/ckpt80.pth --upscale_factor 2
```
Evaluate the model checkpoint *ckpt80.pth* on the Set5 dataset with upscaling factor 2.

### Super-resolve an image
```
usage: super_resolve.py [-h] --input INPUT --model MODEL --output OUTPUT

optional arguments:
  -h, --help       show this help message and exit
  --input INPUT    input image to super resolve
  --model MODEL    model file
  --output OUTPUT  where to save the output image
```
An example to super-resolve an image:
```
$ python super-resolve.py --model checkpoints/ckpt80.pth --input inp.png --output out.png
```

## References
* ["Image Super-Resolution Using Deep Convolutional Networks" - Dong et al.](https://arxiv.org/pdf/1501.00092.pdf).
* [PyTorch example of super-resolution DNN](https://github.com/pytorch/examples/tree/master/super_resolution).
* [Dataset](http://vllab.ucmerced.edu/wlai24/LapSRN).