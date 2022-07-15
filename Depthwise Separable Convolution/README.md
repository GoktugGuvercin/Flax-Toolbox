# Depthwise Separable Convolution

* In standard convolution, small local region of input image in spatial domain is taken to be convolved with the kernel, and this region changes as 
the filter kernel is shifted through image context. However, the extent of connectivity along depth axis is full. In other words, all channels are 
together convolved; hence, the depth of the kernel is equal to the depth of input that will be convolved. 

* Depthwise Separable Convolution, in contrast, split the channels of input into different groups, and  different set of filters are defined with which 
each group will be convolved.  When it is first introduced in Xception paper, each channel in input map is considered to be individual group, and it is 
separately convolved with different kernel, which causes the depth of input and output maps to be same. 

* Depthwise Separable Convolution with group of 1 assumes that cross-channel features are independent from each other, so each channel is convolved with 
different kernel. In other words, only the features lying through the spatial domain in one channel are condensed. Point-wise convolution is like the 
opposite of depth-wise convolution: The size of kernel is 1x1, so the features on spatial extent are ignored, but due to its full depth connectivity the 
features and patterns on different channels would be combined by point-wise convolution.  

* These two types of convolution are complement to each other; they together comprise the effect of standard convolution with fewer number of computations. 
Hence, standard convolution is generally modeled by Depthwise and then pointwise convolution. 

* Depth-wise separable convolution means that each channel or each group of fixed number of channels can be convolved independently from other channels or 
other groups of channels. In this series of convolution operations, one filter or multiple filters for each separate convolution can be used. 
