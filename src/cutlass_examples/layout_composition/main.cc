#include <cstdio>

#include "print_latex.h"

using namespace cute;

template <typename T>
void type(T arg);

int main(int argc, char** argv) {
  // Layout a = make_layout(make_shape (Int<10>{}, Int<2>{}),
  //                        make_stride(Int<16>{}, Int<4>{}));

  // Layout b = make_layout(make_shape (Int< 5>{}, Int<4>{}),
  //                        make_stride(Int< 1>{}, Int<5>{}));
  // Layout c = composition(a, b);
  // print_latex(a);
  // print_latex(b);
  // print_latex(c);

  // (12,(4,8)):(59,(13,1))
  auto a = make_layout(make_shape (12,make_shape ( 4,8)),
                       make_stride(59,make_stride(13,1)));
  // print_latex(a);

  // <3:4, 8:2>
  auto tiler = make_tile(Layout<_3,_4>{},  // Apply 3:4 to mode-0
                         Layout<_8,_2>{}); // Apply 8:2 to mode-1

  // (_3,(2,4)):(236,(26,1))
  auto result = composition(a, tiler);

  TikzColor_BWx8 color;
  auto color_fn = [&](int idx) {
    for (int i = 0; i < size(result); i++) {
      if (idx == result(i)) {
        return color(1);
      }
    }

    return color(0);
  };

  util::fprint_latex(a, color_fn);

  // // Identical to
  // auto same_r = make_layout(composition(layout<0>(a), get<0>(tiler)),
  //                           composition(layout<1>(a), get<1>(tiler)));

  return 0;
}
