#include <cstdio>
#include "cute/layout.hpp"

using namespace cute;

int main(int argc, char** argv) {
  Layout layout = make_layout(
    make_shape(Int<3>{}, Int<5>{}),
    make_stride(Int<1>{}, Int<4>{})
  );
  printf("layout: ");
  print_layout(layout);
  printf("\n");

  // domain: what can go into a function
  printf("%17s", "domain: [");
  for (int i = 0; i < size(layout); i++) {
    if (i == 0) { printf("%d", i); } else { printf(", %2d", i); }
  }
  printf("]\n");

  // range: what actually comes out of a function
  printf("%17s", "range: [");
  for (int i = 0; i < size(layout); i++) {
    if (i == 0) { printf("%d", layout(i)); } else { printf(", %2d", layout(i)); }
  }
  printf("]\n");

  // codomain: what may possibly come out of a function
  printf("%17s", "codomain: [");
  for (int i = layout(0); i < (layout(size(layout) - 1) + 1); i++) {
    if (i == 0) { printf("%d", i); } else { printf(", %2d", i); }
  }
  printf("]\n\n");

  // size and cosize
  printf("%14s: %d\n", "size(layout)", (int)size(layout));
  printf("%14s: %d\n", "cosize(layout)", (int)cosize(layout));
  printf("%14s: %d\n", "cosize(layout)", (int)layout(size(layout) - 1) + 1);

  return 0;
}
