#include <cstdio>

#include "cute/layout.hpp"
#include <cute/tensor.hpp>

#include "cutlass/arch/mma_sm90.h"

#include "util/print_latex.h"

using namespace cute;

template <typename T>
void type(T arg);

int main(int argc, char** argv) {
  // Default stride
  // Layout s2xs4_default = make_layout(Shape<_2, _4>{});
  // printf("s2xs4_default: ");
  // print_layout(s2xs4_default);
  // printf("\n\n");

  auto bM = cute::Int<128>{};
  auto bN = cute::Int<128>{};
  auto bK = cute::Int< 64>{};
  auto bP = cute::Int<  3>{};  // Pipeline

  auto trg_shape = make_shape(bM,bK,bP);
  // type(trg_shape);
  /*
  cute::tuple<
    cute::C<128>, 
    cute::C<64>, 
    cute::C<3> 
  >
  */

  auto atom_layout = GMMA::Layout_MN_SW128_Atom<cute::half_t>{};
  auto block = atom_layout.layout_b();
  // type(block);
  /*
  cute::Layout<
    cute::tuple<
      cute::C<64>, 
      cute::C<8> 
    >, 
    cute::tuple<
      cute::C<1>, 
      cute::C<64> 
    > 
  >
  */
  // util::fprint_latex(block, {}, "block_layout.tex");

  auto tiled_layout = tile_to_shape(block, trg_shape);
  // type(tiled_layout);
  /*
  cute::Layout<
    cute::tuple<
      cute::tuple<
        cute::C<64>, 
        cute::C<2> 
      >, 
      cute::tuple<
        cute::C<8>, 
        cute::C<8> 
      >, 
      cute::C<3> 
    >, 
    cute::tuple<
      cute::tuple<
        cute::C<1>, 
        cute::C<512> 
      >, 
      cute::tuple<
        cute::C<64>, 
        cute::C<1024> 
      >, 
      cute::C<8192> 
    > 
  >
  */

  auto s = make_shape(get<0>(tiled_layout.shape()), get<1>(tiled_layout.shape()));
  auto d = make_stride(get<0>(tiled_layout.stride()), get<1>(tiled_layout.stride()));
  auto result_layout = make_layout(s, d);
  printf("result_layout: "); print(result_layout);
  printf("\n");

  auto rows = coalesce(get<0>(s));
  // type(rows);
  auto cols = coalesce(get<1>(s));
  // type(cols);

  auto coalesced_shape = make_shape(rows, cols);
  // type(coalesced_shape);

  auto coalesced_stride = make_stride(coalesce(get<0>(tiled_layout.stride())), coalesce(get<1>(tiled_layout.stride())));
  // type(coalesced_stride);

  auto block_shape = block.shape();
  auto block_rows = get<0>(block_shape);
  auto block_cols = get<1>(block_shape);

  constexpr auto m = rows / block_rows;
  constexpr auto k = cols / block_cols;

  Layout blocked_layout = make_layout(make_shape(Int<m>{}, Int<k>{}), make_stride(Int<1>{}, Int<m>{}));
  printf("blocked_layout: ");
  print_layout(blocked_layout);
  printf("\n");
  // type(blocked_layout);

  constexpr int C = m * k;
  TikzColor_RGB<C> color;

  auto color_fn = [&](int idx) {
    auto coord = idx2crd(idx, s, d);
    auto coord2d = crd2crd(coord, s, coalesced_shape);
    
    auto i = get<0>(coord2d) / block_rows;
    auto j = get<1>(coord2d) / block_cols;
    auto cidx = blocked_layout(i, j);

    // printf("%d -> ", idx); print(coord2d); printf(" -> (%d,%d) -> %d", i, j, cidx); printf("\n");
    // printf("color = %s\n", color(cidx));
    return color(cidx);
  };

  util::fprint_latex_v2(result_layout, color_fn, "result_layout2.tex");

  return 0;
}
