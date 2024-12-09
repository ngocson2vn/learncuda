#include <cstdio>

#include "cute/layout.hpp"
#include <cute/tensor.hpp>

#include "cutlass/arch/mma_sm90.h"

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

  auto layout = GMMA::Layout_MN_SW128_Atom<cute::half_t>{};
  auto block = layout.layout_b();
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

  return 0;
}
