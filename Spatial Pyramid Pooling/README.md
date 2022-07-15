# Spatial Pyramid Pooling Layer:

* Fully-connected layers at the end of networks as target label predictor need to know how many number of features will be coming from feature maps 
extracted by convolutional backbone. This automatically creates fixed-size constraint on input images fed to the networks. 

* This problem is currently solved by making the network fully-convolutional. In other words, the predictors are designed to be 1x1 convolution layers. 
SPP-Net (2015) comes up with different idea to solve this problem, which is called "Spatial Pyramid Pooling". Spatial pyramid pooling (SPP) is a kind of 
layer placed between convolutional backbone and fully-connected layers. Even if the network is fed by the images of different size, SPP layer always 
produce fixed-size feature vector for FC predictors. In that way, both fixed-size input constraint is removed, but also FC layers become usable as 
predictors. 

* SPP layer at first performs max pooling with a kernel whose size is same as the size of feature maps extracted by convolutional backbone. Then, it 
repeats same operation by halving pooling kernel size in each level. Since the shape of pool kernels is reduced proportional to feature map dimension, 
the number of obtained local receptive bins always remains same regardless of input image size, which guarantees fixed-size feature vector. 
