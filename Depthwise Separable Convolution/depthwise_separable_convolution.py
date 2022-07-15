"""
* In depth-wise separable convolution, the channels of input maps are split into the groups.
* Each channel group is convolved with different kernels separately. 
* How many number of group will exist for those channels is controlled by "feature_group_count" parameter.
* In standard convolution, this parameter is set to 1, which means the channels are not split into separate groups.

* SIMPLE EXAMPLE:

* Input Map: 12x12x64, feature group count = 64, Total number of filters = 128 
* In this case, 64 groups, each of which is composed of 1 channel, are created.
* Each group is convolved by N number of kernels.
* N = 128 / 64 = 2
* Total number of kernels that will be used for the convolution is controlled by the parameter "features" """


import jax
import flax.linen as nn

# random keys for input and parameters are generated
key1, key2 = jax.random.split(jax.random.PRNGKey(seed=23))
input_map = jax.random.normal(key1, (1, 12, 12, 64))

sep_conv = nn.linear.Conv(features=128, kernel_size=(5, 5), feature_group_count=64, padding="VALID")  # depth-wise separable convolution
point_conv = nn.linear.Conv(features=256, kernel_size=(1, 1), feature_group_count=1)  # point-wise (1x1) convolution
model = nn.Sequential([sep_conv, point_conv])

# by model.init(), parameters of the model are randomly generated.
params = model.init(key2, input_map)

# model is applied to input map, and output feature maps are generated
feature_maps = model.apply(params, input_map)

print("Shape of feature maps: ", feature_maps.shape)
