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

  // Default stride
  Layout s2xs4_default = make_layout(Shape<_2, _4>{});
  printf("s2xs4_default: ");
  print_layout(s2xs4_default);
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

  printf("m0: ");
  Layout m0 = layout<0>(s2xs4);
  print1D(m0);
  printf("\n");
  printf("m1: ");
  Layout m1 = layout<1>(s2xs4);
  print1D(m1);
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

  // size - total number of elements
  auto s2xs4_size = size(s2xs4);
  printf("size(s2xs4): %d\n", (int)s2xs4_size);

  printf("\n\n");
  Layout a = Layout<Shape<_4, Shape<_3, _6>>>{};
  printf("Layout<Shape<_4, Shape<_3, _6>>>{}: ");
  print(a);
  printf("\n");
  print_layout(a);
  printf("\n\n");

  Layout s3xs6 = make_layout(
    make_shape(Int<3>{}, Int<6>{}),
    make_stride(Int<4>{}, Int<12>{})
  );
  printf("s3xs6: ");
  print_layout(s3xs6);
  printf("\n\n");

  Layout s32xs8 = make_layout(Shape<_32, _8>{});
  printf("s32xs8: ");
  print_layout(s32xs8);
  printf("\n\n");

  print_latex(s32xs8);

  return 0;
}
