#include <cstdio>

#include "cute/layout.hpp"

#include "util/print_latex.h"

using namespace cute;

template <typename T>
void type(T arg);

int main(int argc, char** argv) {
  Layout block = make_layout(Shape<_3, _4>{}, Stride<_1, _3>{});
  auto target_shape = Shape<_9, _12>{};
  auto result_layout = tile_to_shape(block, target_shape);
  printf("result_layout: ");
  print_layout(result_layout);
  printf("\n");

  auto rows = coalesce(get<0>(result_layout.shape()));
  auto cols = coalesce(get<1>(result_layout.shape()));
  auto coalesced_shape = make_shape(rows, cols);

  auto block_rows = get<0>(block.shape());
  auto block_cols = get<1>(block.shape());

  constexpr auto m = rows / block_rows;
  constexpr auto k = cols / block_cols;

  Layout blocked_layout = make_layout(make_shape(Int<m>{}, Int<k>{}), make_stride(Int<1>{}, Int<m>{}));
  printf("blocked_layout: ");
  print_layout(blocked_layout);
  printf("\n");

  constexpr int C = m * k;
  TikzColor_RGB<C> color;

  auto color_fn = [&](int idx) {
    auto coord = idx2crd(idx, result_layout.shape(), result_layout.stride());
    auto coord2d = crd2crd(coord, result_layout.shape(), coalesced_shape);
    
    auto i = get<0>(coord2d) / block_rows;
    auto j = get<1>(coord2d) / block_cols;
    auto cidx = blocked_layout(i, j);

    return color(cidx);
  };

  util::fprint_latex_v2(result_layout, color_fn, "result_layout1.tex");

  return 0;
}
