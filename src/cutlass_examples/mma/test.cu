#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <cute/tensor.hpp>

#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/arch/mma_sm90.h"

using namespace cute;
using TA = cute::half_t;

template <typename T>
class TD;

int main(int argc, char** argv) {
  auto bM = cute::Int<128>{};
  auto bN = cute::Int<128>{};
  auto bK = cute::Int< 64>{};
  auto bP = cute::Int<  3>{};  // Pipeline

  // TD<GMMA::Layout_MN_SW128_Atom<TA>> td;

  // Define the smem layouts (static)
  /*
  GMMA::Layout_MN_SW128_Atom<TA> expands to
  using cute::SM90::GMMA::Layout_MN_SW128_Atom<TA> = cute::ComposedLayout<cute::Swizzle<3, 4, 3>, cute::smem_ptr_flag_bits<16>, cute::Layout<cute::Shape<cute::_64, cute::_8>, cute::Stride<cute::_1, cute::_64>>> 
  */
  auto sA = cute::tile_to_shape(GMMA::Layout_MN_SW128_Atom<TA>{}, cute::make_shape(bM,bK,bP));
  cute::print(sA);
  printf("\n");
}
