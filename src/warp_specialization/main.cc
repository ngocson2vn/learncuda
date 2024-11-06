#include <cstdlib>
#include <cstdio>
#include "ws.h"

int main() {
  int N = kWarpSize * 2 * 4;
  float* h_in = new float[N];
  for (size_t i = 0; i < N; i++) {
    h_in[i] = i;
  }

  printf("in:");
  for (size_t i = 0; i < N; i++) {
    printf(" %.1f", h_in[i]);
  }
  printf("\n\n");

  // Process data
  float* h_out = new float[N];
  process(h_in, h_out, N);

  printf("out:");
  for (size_t i = 0; i < N; i++) {
    printf(" %.1f", h_out[i]);
  }
  printf("\n");

  free(h_in);
  free(h_out);

  return 0;
}
