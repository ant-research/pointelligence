#include "common.h"

namespace sparse_engines_cuda {
// http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
unsigned int next_highest_power_of_2(unsigned int v) {
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}
} // namespace sparse_engines_cuda