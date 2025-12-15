#include <cstdio>
#include <iostream>
#include <bitset>

#include <cute/config.hpp>                      // CUTE_HOST_DEVICE
#include <cute/container/tuple.hpp>             // cute::is_tuple
#include <cute/numeric/integral_constant.hpp>   // cute::constant
#include <cute/numeric/math.hpp>                // cute::max, cute::min

namespace sony {

template<typename T> 
class TD;

// 
// https://github.com/NVIDIA/cutlass/blob/main/include/cute/swizzle.hpp
// 
// A generic Swizzle functor
// Given an offset with binary representation:
/* 0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
 *                               ^--^ MBase is the number of least-sig bits to keep constant
 *                  ^-^       ^-^     BBits is the number of bits in the mask
 *                    ^---------^     SShift is the distance to shift the YYY mask
 *                                       (pos shifts YYY to the right, neg shifts YYY to the left)
 *
 * e.g. Given
 * 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx
 * the result is
 * 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx where AA = ZZ xor YY
 */
template <int BBits, int MBase, int SShift = BBits>
struct Swizzle
{
  static constexpr int num_bits = BBits;
  static constexpr int num_base = MBase;
  static constexpr int num_shft = SShift;

  static_assert(num_base >= 0,             "MBase must be positive.");
  static_assert(num_bits >= 0,             "BBits must be positive.");
  static_assert(abs(num_shft) >= num_bits, "abs(SShift) must be more than BBits.");

  // using 'int' type here to avoid unintentially casting to unsigned... unsure.
  using bit_msk = cute::constant<int, (1 << num_bits) - 1>; // cute::C<(1 << num_bits) - 1>
  // TD<bit_msk> t;

  using yyy_msk = cute::constant<int, bit_msk{} << (num_base + cute::max(0,num_shft))>;
  using zzz_msk = cute::constant<int, bit_msk{} << (num_base - cute::min(0,num_shft))>;
  using msk_sft = cute::constant<int, num_shft>;

  static constexpr uint32_t swizzle_code = uint32_t(yyy_msk{} | zzz_msk{});

  template <class Offset>
  CUTE_HOST_DEVICE constexpr static
  auto
  apply(Offset const& offset)
  {
    // YYY = offset & yyy_msk{}
    // offset ^ (YYY >> msk_sft{}) <=> ZZZ ^ YYY, i.e., the original YYY bits flips ZZZ bits if Y = 1
    return offset ^ cute::shiftr(offset & yyy_msk{}, msk_sft{});   // ZZZ ^= YYY
  }

  template <class Offset>
  CUTE_HOST_DEVICE constexpr
  auto
  operator()(Offset const& offset) const
  {
    return apply(offset);
  }

  template <int B, int M, int S>
  CUTE_HOST_DEVICE constexpr
  auto
  operator==(Swizzle<B,M,S> const&) const
  {
    return B == BBits && M == MBase && S == SShift;
  }
};

} // namespace sony

int main(int argc, char** argv) {
  unsigned short offset = 65;
  std::cout << "offset = " << std::bitset<16>(offset) << std::endl;

  constexpr int num_bits = 5;
  std::cout << "(1 << num_bits) = " << std::bitset<16>(1 << num_bits) << std::endl;
  std::cout << "bit_msk = " << std::bitset<16>((1 << num_bits) - 1) << std::endl;

  constexpr int num_base = 0;
  constexpr int num_shft = 6;
  std::cout << "yyy_msk = " << std::bitset<16>(((1 << num_bits) - 1) << num_base + num_shft) << std::endl;
  std::cout << "zzz_msk = " << std::bitset<16>(((1 << num_bits) - 1) << num_base - 0) << std::endl;
  std::cout << "msk_sft = " << num_shft << std::endl;

  sony::Swizzle<num_bits, num_base, num_shft> swizzle;
  std::cout << "swizzle(offset) = " << swizzle(offset) << std::endl;
}
