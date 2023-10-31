#include "headers/weights.h"
#include "headers/defines.h"
#include "headers/activations.h"
#include <hls_video.h>

#include "ap_fixed.h"

#define TOTAL_WIDTH	16
#define INT_WIDTH	5

typedef ap_fixed<TOTAL_WIDTH, INT_WIDTH> float16_t;

struct float16_t_tlast {
	float16_t value;
	int last = 0;
};

#define CONV1_LINE_BUFFER_SIZE (IMAGE_SIZE * (CONV1_KERNEL_SIZE-1) + CONV1_KERNEL_SIZE)

void conv_layer1(hls::stream<float16_t> &in, hls::stream<float16_t> &out,
		float16_t weight[CONV1_KERNEL_SIZE][CONV1_KERNEL_SIZE][1][CONV1_FILTERS],
		float16_t bias[CONV1_BIAS_SIZE]) {
	int i, j, k, filter;
	float16_t sum, pixel;
	int row, col;
	hls::LineBuffer<CONV1_LINE_BUFFER_SIZE, 1, float16_t> conv_buff;
	
	/*
	 * Read the initial buffer
	 * */

	for (i = 0; i < CONV1_LINE_BUFFER_SIZE; i++) {
		if (in.empty() == 0) {
			in >> pixel;
			conv_buff.shift_up(0);
			conv_buff.insert_top(pixel, 0);
		}
	}

	for (i = 0; i < (IMAGE_SIZE - CONV1_KERNEL_SIZE + 1); i += 1) {
		for (j = 0; j < (IMAGE_SIZE - CONV1_KERNEL_SIZE + 1); j += 1) {
			for (filter = 0; filter < CONV1_FILTERS; filter++) {
				sum = 0;
				for (row = 0; row < CONV1_KERNEL_SIZE; row++) {
					for (col = 0; col < CONV1_KERNEL_SIZE; col++) {
							int kernel_row_idx, kernel_col_idx;
							static float16_t x, w;
							kernel_row_idx = row * IMAGE_SIZE;
							kernel_col_idx = col;
							x = conv_buff.getval(kernel_row_idx + kernel_col_idx,0);
							w = weight[row][col][0][filter];
							sum += x * w;
						}
				}
				out << relu(sum + bias[filter]);
			}

			if ((j + 1 < (IMAGE_SIZE - CONV1_KERNEL_SIZE + 1))) {
				if (in.empty() == 0) {
					in >> pixel;
					conv_buff.shift_up(0);
					conv_buff.insert_top(pixel, 0);
				}
			} else if ((i + 1 < (IMAGE_SIZE - CONV1_KERNEL_SIZE + 1)) && (j + 1 >= (IMAGE_SIZE - CONV1_KERNEL_SIZE + 1))) {
				for (int k = 0; k < CONV1_KERNEL_SIZE; k++) {
					if (in.empty() == 0) {
						in >> pixel;
						conv_buff.shift_up(0);
						conv_buff.insert_top(pixel, 0);
					}
				}
			}
		}
	}
}

#define POOL1_LINE_BUFFER_SIZE (POOL1_SIZE*POOL1_CHANNELS)

void pool_layer1(hls::stream<float16_t> &in, hls::stream<float16_t> &out) {
	int i, j, k, l, m;
	float16_t feature;
	hls::LineBuffer<POOL1_LINE_BUFFER_SIZE, 1, float16_t> pool_buff;

	for (i = 0; i < POOL1_SIZE; i++) {
		for (l = 0; l < POOL1_KERNEL_SIZE; l++) {
			for (j = 0; j < POOL1_SIZE; j++) {
				for (m = 0; m < POOL1_KERNEL_SIZE; m++) {
					for (k = 0; k < POOL1_CHANNELS; k++) {
						in >> feature;
						if (l == 0 && m == 0)
							pool_buff.val[j * POOL1_CHANNELS + k][0] = feature;
						else
							pool_buff.val[j * POOL1_CHANNELS + k][0] = pool_buff.val[j * POOL1_CHANNELS + k][0] > feature ? pool_buff.val[j * POOL1_CHANNELS + k][0] : feature;
						if (l == (POOL1_KERNEL_SIZE - 1) && m == (POOL1_KERNEL_SIZE - 1))
							out << pool_buff.val[j * POOL1_CHANNELS + k][0];
					}
				}
			}
			for (int skip = POOL1_SIZE * POOL1_STRIDE; skip < POOL1_INPUT_FEATURE_SIZE; skip++) {
				for (int channel = 0; channel < POOL1_CHANNELS; channel++) {
					in >> feature;
				}
			}
		}
	}
	for (int skip_row = POOL1_SIZE * POOL1_STRIDE; skip_row < POOL1_INPUT_FEATURE_SIZE; skip_row++) {
		for (int skip_col = 0; skip_col < POOL1_INPUT_FEATURE_SIZE; skip_col++) {
			for (int skip_channel = 0; skip_channel < POOL1_INPUT_FEATURE_CHANNELS; skip_channel++) {
				in >> feature;
			}
		}
	}
}

#define CONV2_LINE_BUFFER_SIZE (POOL1_SIZE * POOL1_CHANNELS * (CONV2_KERNEL_SIZE-1) + CONV2_KERNEL_SIZE * POOL1_CHANNELS)

void conv_layer2(hls::stream<float16_t> &in, hls::stream<float16_t> &out,
		float16_t weight[CONV2_KERNEL_SIZE][CONV2_KERNEL_SIZE][CONV2_CHANNELS][CONV2_FILTERS],
		float16_t bias[CONV2_BIAS_SIZE]) {
	int i, j, k, filter;
	float16_t sum, feature;
	int row, col, channel;
	hls::LineBuffer<CONV2_LINE_BUFFER_SIZE, 1, float16_t> conv_buff;
	
	/*
	 * Read the initial buffer
	 * */

	for (i = 0; i < CONV2_LINE_BUFFER_SIZE; i++) {
		if (in.empty() == 0) {
			in >> feature;
			conv_buff.shift_up(0);
			conv_buff.insert_top(feature, 0);
		}
	}

	for (i = 0; i < (POOL1_SIZE - CONV2_KERNEL_SIZE + 1); i += 1) {
		for (j = 0; j < (POOL1_SIZE - CONV2_KERNEL_SIZE + 1); j += 1) {
			for (filter = 0; filter < CONV2_FILTERS; filter++) {
				sum = 0;
				for (row = 0; row < CONV2_KERNEL_SIZE; row++) {
					for (col = 0; col < CONV2_KERNEL_SIZE; col++) {
						for (channel = 0; channel < CONV2_CHANNELS; channel++) {
							int kernel_row_idx, kernel_col_idx;
							static float16_t x, w;
							kernel_row_idx = row * POOL1_SIZE * POOL1_CHANNELS;
							kernel_col_idx = col * POOL1_CHANNELS;
							x = conv_buff.getval(kernel_row_idx + kernel_col_idx + channel,0);
							w = weight[row][col][channel][filter];
							sum += x * w;
						}
					}
				}
				out << relu(sum + bias[filter]);
			}

			if ((j + 1 < (POOL1_SIZE - CONV2_KERNEL_SIZE + 1))) {
				for (int k = 0; k < POOL1_CHANNELS; k++) {
					if (in.empty() == 0) {
						in >> feature;
						conv_buff.shift_up(0);
						conv_buff.insert_top(feature, 0);
					}
				}
			} else if ((i + 1 < (POOL1_SIZE - CONV2_KERNEL_SIZE + 1)) && (j + 1 >= (POOL1_SIZE - CONV2_KERNEL_SIZE + 1))) {
				for (int k = 0; k < CONV2_KERNEL_SIZE * POOL1_CHANNELS; k++) {
					if (in.empty() == 0) {
						in >> feature;
						conv_buff.shift_up(0);
						conv_buff.insert_top(feature, 0);
					}
				}
			}
		}
	}
}

#define POOL2_LINE_BUFFER_SIZE (POOL2_SIZE*POOL2_CHANNELS)

void pool_layer2(hls::stream<float16_t> &in, hls::stream<float16_t> &out) {
	int i, j, k, l, m;
	float16_t feature;
	hls::LineBuffer<POOL2_LINE_BUFFER_SIZE, 1, float16_t> pool_buff;

	for (i = 0; i < POOL2_SIZE; i++) {
		for (l = 0; l < POOL2_KERNEL_SIZE; l++) {
			for (j = 0; j < POOL2_SIZE; j++) {
				for (m = 0; m < POOL2_KERNEL_SIZE; m++) {
					for (k = 0; k < POOL2_CHANNELS; k++) {
						in >> feature;
						if (l == 0 && m == 0)
							pool_buff.val[j * POOL2_CHANNELS + k][0] = feature;
						else
							pool_buff.val[j * POOL2_CHANNELS + k][0] = pool_buff.val[j * POOL2_CHANNELS + k][0] > feature ? pool_buff.val[j * POOL2_CHANNELS + k][0] : feature;
						if (l == (POOL2_KERNEL_SIZE - 1) && m == (POOL2_KERNEL_SIZE - 1))
							out << pool_buff.val[j * POOL2_CHANNELS + k][0];
					}
				}
			}
			for (int skip = POOL2_SIZE * POOL2_STRIDE; skip < POOL2_INPUT_FEATURE_SIZE; skip++) {
				for (int channel = 0; channel < POOL2_CHANNELS; channel++) {
					in >> feature;
				}
			}
		}
	}
	for (int skip_row = POOL2_SIZE * POOL2_STRIDE; skip_row < POOL2_INPUT_FEATURE_SIZE; skip_row++) {
		for (int skip_col = 0; skip_col < POOL2_INPUT_FEATURE_SIZE; skip_col++) {
			for (int skip_channel = 0; skip_channel < POOL2_INPUT_FEATURE_CHANNELS; skip_channel++) {
				in >> feature;
			}
		}
	}
}

#define CONV3_LINE_BUFFER_SIZE (POOL2_SIZE * POOL2_CHANNELS * (CONV3_KERNEL_SIZE-1) + CONV3_KERNEL_SIZE * POOL2_CHANNELS)

void conv_layer3(hls::stream<float16_t> &in, hls::stream<float16_t> &out,
		float16_t weight[CONV3_KERNEL_SIZE][CONV3_KERNEL_SIZE][CONV3_CHANNELS][CONV3_FILTERS],
		float16_t bias[CONV3_BIAS_SIZE]) {
	int i, j, k, filter;
	float16_t sum, feature;
	int row, col, channel;
	hls::LineBuffer<CONV3_LINE_BUFFER_SIZE, 1, float16_t> conv_buff;

	/*
	 * Read the initial buffer
	 * */

	for (i = 0; i < CONV3_LINE_BUFFER_SIZE; i++) {
		if (in.empty() == 0) {
			in >> feature;
			conv_buff.shift_up(0);
			conv_buff.insert_top(feature, 0);
		}
	}

	for (i = 0; i < (POOL2_SIZE - CONV3_KERNEL_SIZE + 1); i += 1) {
		for (j = 0; j < (POOL2_SIZE - CONV3_KERNEL_SIZE + 1); j += 1) {
			for (filter = 0; filter < CONV3_FILTERS; filter++) {
				sum = 0;
				for (row = 0; row < CONV3_KERNEL_SIZE; row++) {
					for (col = 0; col < CONV3_KERNEL_SIZE; col++) {
						for (channel = 0; channel < CONV3_CHANNELS; channel++) {
							int kernel_row_idx, kernel_col_idx;
							static float16_t x, w;
							kernel_row_idx = row * POOL2_SIZE * POOL2_CHANNELS;
							kernel_col_idx = col * POOL2_CHANNELS;
							x = conv_buff.getval(kernel_row_idx + kernel_col_idx + channel,0);
							w = weight[row][col][channel][filter];
							sum += x * w;
						}
					}
				}
				out << relu(sum + bias[filter]);
			}

			if ((j + 1 < (POOL2_SIZE - CONV3_KERNEL_SIZE + 1))) {
				for (int k = 0; k < POOL2_CHANNELS; k++) {
					if (in.empty() == 0) {
						in >> feature;
						conv_buff.shift_up(0);
						conv_buff.insert_top(feature, 0);
					}
				}
			} else if ((i + 1 < (POOL2_SIZE - CONV3_KERNEL_SIZE + 1)) && (j + 1 >= (POOL2_SIZE - CONV3_KERNEL_SIZE + 1))) {
				for (int k = 0; k < CONV3_KERNEL_SIZE * POOL2_CHANNELS; k++) {
					if (in.empty() == 0) {
						in >> feature;
						conv_buff.shift_up(0);
						conv_buff.insert_top(feature, 0);
					}
				}
			}
		}
	}
}

void fullyconnected_layer(hls::stream<float16_t> &in, hls::stream<float16_t_tlast> &out,
		float16_t weight[FULLYCONNECTED_WEIGHTS][1],
		float16_t bias[FULLYCONNECTED_BIAS_SIZE]) {
	float16_t feature;
	float16_t output;
	float16_t_tlast s;
	s.last = 0;
	
	in >> feature;
	output = weight[0][0] * feature;

	for (int i = 1; i < FULLYCONNECTED_WEIGHTS; i++) {
		in >> feature;
		output += weight[i][0] * feature;
	}
	s.value = sigmoid(output + bias[0]);
	if (s.value > 0)
		s.last = 1;
	else
		s.last = 0;
	out.write(s);
}

void cnn(hls::stream<float16_t> &image_in, hls::stream<float16_t_tlast> &out) {
	#pragma HLS INTERFACE axis port=image_in
	#pragma HLS INTERFACE axis port=out
	#pragma HLS INTERFACE s_axilite port=return
	#pragma HLS DATAFLOW disable_start_propagation
	static hls::stream<float16_t> conv1_out;
	static hls::stream<float16_t> conv2_out;
    static hls::stream<float16_t> conv3_out;
	static hls::stream<float16_t> pool1_out;
	static hls::stream<float16_t> pool2_out;

	conv_layer1(image_in, conv1_out, conv2d_layer1_weights, conv2d_layer1_biases);
	pool_layer1(conv1_out, pool1_out);

    conv_layer2(pool1_out, conv2_out, conv2d_layer2_weights, conv2d_layer2_biases);
	pool_layer2(conv2_out, pool2_out);

	conv_layer3(pool2_out, conv3_out, conv2d_layer3_weights, conv2d_layer3_biases);

	fullyconnected_layer(conv3_out, out, fc_layer_weights, fc_layer_biases);
}
