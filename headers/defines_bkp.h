#define IMAGE_SIZE  64
// #define IMAGE_CHANNELS		1

#define CONV1_KERNEL_SIZE 	3
#define CONV1_CHANNELS 		1
#define CONV1_FILTERS 		8
#define CONV1_BIAS_SIZE 	8
// #define CONV1_STRIDE 		1

#define CONV2_KERNEL_SIZE 	3
#define CONV2_CHANNELS 		8
#define CONV2_FILTERS 		8
#define CONV2_BIAS_SIZE		8
// #define CONV2_STRIDE 		1

#define POOL1_SIZE          30
#define POOL1_CHANNELS      8
#define POOL1_KERNEL_SIZE   2
#define POOL1_STRIDE        2

#define POOL1_INPUT_FEATURE_SIZE        60
#define POOL1_INPUT_FEATURE_CHANNELS    8

#define CONV3_KERNEL_SIZE 	3
#define CONV3_CHANNELS 		8
#define CONV3_FILTERS 		16
#define CONV3_BIAS_SIZE		16
#define CONV3_STRIDE 		1

#define CONV4_KERNEL_SIZE 	3
#define CONV4_CHANNELS 		16
#define CONV4_FILTERS 		16
#define CONV4_BIAS_SIZE		16
#define CONV4_STRIDE 		1

#define POOL2_SIZE          13
#define POOL2_CHANNELS      16
#define POOL2_KERNEL_SIZE   2
#define POOL2_STRIDE        2

#define POOL2_INPUT_FEATURE_SIZE        26
#define POOL2_INPUT_FEATURE_CHANNELS    16

#define CONV5_KERNEL_SIZE 	3
#define CONV5_CHANNELS 		16
#define CONV5_FILTERS 		16
#define CONV5_BIAS_SIZE		16
#define CONV5_STRIDE 		1

#define CONV6_KERNEL_SIZE 	3
#define CONV6_CHANNELS 		16
#define CONV6_FILTERS 		16
#define CONV6_BIAS_SIZE		16
#define CONV6_STRIDE 		1

#define POOL3_SIZE          4
#define POOL3_CHANNELS      16
#define POOL3_KERNEL_SIZE   2
#define POOL3_STRIDE        2

#define POOL3_INPUT_FEATURE_SIZE        9
#define POOL3_INPUT_FEATURE_CHANNELS    16

#define CONV7_KERNEL_SIZE 	3
#define CONV7_CHANNELS 		16
#define CONV7_FILTERS 		16
#define CONV7_BIAS_SIZE		16
#define CONV7_STRIDE 		1

// #define FLATTEN_SIZE		1152

#define FULLYCONNECTED_WEIGHTS		64
// #define FULLYCONNECTED_WEIGHTS_W		10
#define FULLYCONNECTED_BIAS_SIZE		1

// #define FULLYCONNECTED_ACT_SIZE		1