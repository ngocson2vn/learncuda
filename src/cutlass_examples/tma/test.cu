#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <cute/atom/copy_atom.hpp>

using namespace cute;

int main(int argc, char** argv) {
  Copy_Atom<Copy_Traits<cute::SM90_TMA_LOAD, cute::C<524288>, cute::AuxTmaParams<cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0> >, const cute::Layout<cute::tuple<cute::C<128>, cute::C<128> >, cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0> > >&, const cute::Swizzle<0, 4, 3>&>>, float> atom;

  uint64_t n = 0;
  atom.with(n);
}