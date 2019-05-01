# Collection of Super-Resolution models
This repo contains my implementation of many well-known super-resolution deep neural network (DNN) models. Each folder contains a separate model along with instruction on how to train, evaluate, and super-resolve an input image. This page acts as a summary on what models I have implemented and what I plan to implement next.

This is a just-for-fun project.

## Models
The following models have been implemented:
* SRCNN - Super-Resolution Convolutional Neural Network: [paper](https://arxiv.org/pdf/1501.00092.pdf).
* ESPCN - Efficient Sub-Pixel Convolutional Neural Network: [paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Real-Time_Single_Image_CVPR_2016_paper.pdf).
* FSRCNN - Accelerating the Super-Resolution Convolutional Neural Network: [paper](https://arxiv.org/pdf/1608.00367.pdf).

## Performance

Performance of my implementations (versus results reported by authors which are put in parentheses).

**PSNR**

| Algorithm | Scale | Set5 | Set14 | BSD100 |
| --------- |:-----:|:----:|:-----:|:------:|
| SRCNN<br>ESPCN<br>FSRCNN | 2x | 36.47 (36.66)<br>x<br>37.04 (37.00) | 32.19 (32.45)<br>x<br>32.57 (32.63) | 31.21<br>x<br>31.53 |
| SRCNN<br>ESPCN<br>FSRCNN | 3x | 32.54 (32.75)<br>x<br>33.15 (33.16) | 29.06 (29.30)<br>x<br>29.39 (29.43) | 28.28<br>x<br>28.52 (28.60) |
| SRCNN<br>ESPCN<br>FSRCNN | 4x | 30.27 (30.49)<br>x<br>30.86 | 27.29 (27.50)<br>x<br>27.66 | 26.78<br>x<br>27.02 |
