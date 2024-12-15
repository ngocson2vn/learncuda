#include <cstdio>
#include "cute/layout.hpp"

using namespace cute;

int main(int argc, char** argv) {
  Layout A = make_layout(Shape<_2, _2>{}, Stride<_1, _6>{});
  printf("A: ");
  print_layout(A);
  printf("\n\n");

  Layout B = complement(A, 24);
  printf("B: ");
  print_layout(B);
  printf("\n\n");

  Layout result = make_layout(A, B);
  printf("result: ");
  print_layout(result);
  printf("\n\n");
  printf("cosize: %d\n\n", (int)cosize(result));

  return 0;
}
