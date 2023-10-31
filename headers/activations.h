#include "ap_fixed.h"
#include <math.h>

#define TOTAL_WIDTH	16
#define INT_WIDTH	5

typedef ap_fixed<TOTAL_WIDTH, INT_WIDTH> float16_t;

float16_t relu(float16_t x)
{
	return x > (float16_t)0 ? x : (float16_t)0;
}

float16_t sigmoid(float16_t x)
{
	return (float16_t)(1.0 / (1.0 + exp(-(float)x)));
}