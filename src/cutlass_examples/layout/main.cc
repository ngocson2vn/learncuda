#include <cstdio>
#include "cute/layout.hpp"

using namespace cute;

template <class Shape, class Stride>
void print1D(Layout<Shape,Stride> const& layout) {
  for (int i = 0; i < size(layout); ++i) {
    printf("%3d  ", layout(i));
  }
}

template <class Shape, class Stride>
void print2D(Layout<Shape,Stride> const& layout) {
  for (int m = 0; m < size<0>(layout); ++m) {
    for (int n = 0; n < size<1>(layout); ++n) {
      printf("layout(%d, %d): %3d  ", m, n, layout(m,n));
    }
    printf("\n");
  }
}

int main(int argc, char** argv) {
  printf("\n\n");
  Layout s8 = make_layout(Int<8>{});
  printf("s8: ");
  print(s8);
  printf("\n\n");

  Layout s2xs4 = make_layout(
    make_shape(Int<2>{}, Int<4>{}),
    make_stride(Int<2>{}, Int<2>{})
  );
  printf("size(s2xs4): %d\n", (int)size(s2xs4));
  printf("s2xs4: ");
  print_layout(s2xs4);
  printf("\n\n");
  printf("1D(s2xs4): ");
  print1D(s2xs4);
  printf("\n\n");
  print2D(s2xs4);
  printf("\n\n");

  Layout s2xd4_a = make_layout(
    make_shape(Int<2>{}, 4),
    make_stride(Int<12>{}, Int<1>{})
  );
  printf("s2xd4_a: ");
  print_layout(s2xd4_a);
  printf("\n\n");
  print2D(s2xd4_a);
  printf("\n\n");

  Layout s2xh4 = make_layout(
    make_shape(2, make_shape(2, 2)),
    make_stride(4, make_stride(2, 1))
  );
  printf("s2xh4: ");
  print_layout(s2xh4);
  printf("\n\n");
  print2D(s2xh4);
  printf("\n\n");

  return 0;
}
