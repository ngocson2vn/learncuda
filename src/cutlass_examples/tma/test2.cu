#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <cute/atom/copy_atom.hpp>

using namespace cute;

template<typename T>
void type(T arg);

template <class... Args>
struct Test_Traits;

template <class DataType>
struct Test_Traits<SM90_TMA_LOAD_OP, DataType> {
  tuple<DataType> data;
};

template <class... Args>
struct Test_Traits<cute::SM90_TMA_LOAD, Args...> {
  constexpr
  Copy_Traits<SM90_TMA_LOAD_OP, int>
  with(
    uint64_t& tma_mbar,
    [[maybe_unused]] uint16_t const& multicast_mask = 0,
    TMA::CacheHintSm90 const& cache_hint = TMA::CacheHintSm90::EVICT_NORMAL) const {
    // We accept multicast_mask here to keep the API for both atoms consistent
    printf("Test_Traits with\n\n");
    printf("with %s\n\n", __PRETTY_FUNCTION__);
    return {{}, {nullptr, nullptr, static_cast<uint64_t>(0)}};
    // return {{}, {nullptr, nullptr, static_cast<uint64_t>(0)}};
  }
};

template <class... Args>
struct Test_Atom;

template <class... Args, class CopyInternalType>
struct Test_Atom<Test_Traits<Args...>, CopyInternalType>
  : Test_Traits<Args...>
{
  using Traits = Test_Traits<Args...>;

  // Additional Trait parameters/transformations
  template <class... TraitsArgs>
  CUTE_HOST_DEVICE
  auto
  with(TraitsArgs&&... args) const {
    printf("with %s\n\n", __PRETTY_FUNCTION__);

    // Traits is the alias of the base class
    // call base class's method
    auto traits = Traits::with(static_cast<TraitsArgs&&>(args)...);
    printf("traits: %p\n\n", &traits);
  }
};

int main(int argc, char** argv) {
  Test_Atom<Test_Traits<cute::SM90_TMA_LOAD, cute::C<524288>, cute::AuxTmaParams<cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0> >, const cute::Layout<cute::tuple<cute::C<128>, cute::C<128> >, cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0> > >&, const cute::Swizzle<0, 4, 3>&>>, float> atom;
  uint64_t mbarrier = 100;
  atom.with(mbarrier);
}