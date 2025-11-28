#include <memory>

#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>

using namespace cute;

namespace sony {
  template <class T, class = void>
  struct iter_ref { using type = decltype(*declval<T&>()); };

  template <class T>
  struct iter_ref<T, void_t<typename T::reference>> { using type = typename T::reference; };
}

int main(int argc, char** argv) {
  // 
  // tensor A
  // 
  // cute::Tensor<
  //   cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>, 
  //   cute::Layout<
  //     cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>, 
  //     cute::tuple<cute::C<0>, cute::C<64>, cute::C<256>>
  //   >
  // >

  // 
  // smem_tensor
  // 
  // cute::Tensor<
  //   cute::ViewEngine<cute::swizzle_ptr<cute::Swizzle<3, 4, 3>, cute::smem_ptr<cutlass::bfloat16_t *>>>, 
  //   cute::Layout<
  //     cute::tuple<cute::C<64>, cute::tuple<cute::C<8>, cute::C<2>>>, 
  //     cute::tuple<cute::C<1>, cute::tuple<cute::C<64>, cute::C<1024>>>
  //   >
  // >

  // 
  // Create smem_tensor
  // 
  std::unique_ptr<cutlass::bfloat16_t> smem_uptr(new cutlass::bfloat16_t(0));
  PRINT_EXPR_TYPE("raw_smem_ptr type", smem_uptr.get());
  printf("raw_smem_ptr: %p\n", smem_uptr.get());

  auto smem_ptr = cute::make_smem_ptr(smem_uptr.get());

  auto swizzle_ptr = cute::make_swizzle_ptr(smem_ptr, cute::Swizzle<3, 4, 3>{});
  PRINT_EXPR_TYPE("swizzle_ptr", swizzle_ptr);
  // swizzle_ptr: T = cute::swizzle_ptr<cute::Swizzle<3, 4, 3>, cute::smem_ptr<cutlass::bfloat16_t*> >

  // 
  // Iterator = cute::swizzle_ptr<cute::Swizzle<3, 4, 3>, cute::smem_ptr<cutlass::bfloat16_t*> >
  // 
  cute::ViewEngine<decltype(swizzle_ptr)> ve{swizzle_ptr};
  PRINT_EXPR_TYPE("ve", ve);
  // ve: T = cute::ViewEngine<cute::swizzle_ptr<cute::Swizzle<3, 4, 3>, cute::smem_ptr<cutlass::bfloat16_t*> > >

  auto smem_tensor_layout = cute::Layout<
    cute::tuple<cute::C<64>, cute::tuple<cute::C<8>, cute::C<2>>>, 
    cute::tuple<cute::C<1>, cute::tuple<cute::C<64>, cute::C<1024>>>
  >{};

  // 
  // ve.begin()
  // 
  PRINT_EXPR_TYPE("ve.begin() type:", ve.begin());
  // ve.begin() type = swizzle_ptr<cute::Swizzle<3, 4, 3>, cute::smem_ptr<cutlass::bfloat16_t *>> &

  // ParamType       = Iterator const&
  // ve.begin() type = swizzle_ptr<cute::Swizzle<3, 4, 3>, cute::smem_ptr<cutlass::bfloat16_t *>> &
  // ==> Iterator = swizzle_ptr<cute::Swizzle<3, 4, 3>, cute::smem_ptr<cutlass::bfloat16_t *>>
  // Inside cute::make_tensor(), it will re-create a cute::ViewEngine<cute::swizzle_ptr<cute::Swizzle<3, 4, 3>, cute::smem_ptr<cutlass::bfloat16_t *>>> object 
  // from ve.begin()
  auto smem_tensor = cute::make_tensor(ve.begin(), smem_tensor_layout);
  PRINT_EXPR_TYPE("smem_tensor", smem_tensor);
  /*
  smem_tensor:
  cute::Tensor<
    cute::ViewEngine<cute::swizzle_ptr<cute::Swizzle<3, 4, 3>, cute::smem_ptr<cutlass::bfloat16_t*> > >, 
    cute::Layout<
      cute::tuple<cute::C<64>, cute::tuple<cute::C<8>, cute::C<2> > >, 
      cute::tuple<cute::C<1>, cute::tuple<cute::C<64>, cute::C<1024> > > 
    > 
  >
  */

  // smem_tensor.data() -> engine().begin() -> ViewEngine::begin() -> a reference to Iterator object
  PRINT_EXPR_TYPE("smem_tensor.data() type", smem_tensor.data());
  // smem_tensor.data() type: swizzle_ptr<cute::Swizzle<3, 4, 3>, cute::smem_ptr<cutlass::bfloat16_t *>> &

  // Get raw pointer
  auto raw_smem_ptr = raw_pointer_cast(smem_tensor.data());
  PRINT_EXPR_TYPE("raw_smem_ptr type", raw_smem_ptr);
  printf("raw_smem_ptr: %p\n", raw_smem_ptr);

  // 
  // Create a GmmaDescriptor
  // 
  // make_gmma_desc() does the following steps:
  //   - extracts raw smem ptr from smem_tensor -> start_address
  //   - computes leading_byte_offset and stride_byte_offset from the layout of smem_tensor
  auto gmma_desc = cute::SM90::GMMA::make_gmma_desc<cute::SM90::GMMA::Major::MN>(smem_tensor);

  // 
  // Create a DescriptorIterator
  // 
  // Later, tensor slicing operations will rely on this DescriptorIterator object 
  // to create different GmmaDescriptor objects
  auto descIterator = cute::SM90::GMMA::DescriptorIterator{gmma_desc};

  // 
  // Create tensor A
  // 
  auto layoutA = cute::Layout<
    cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>, cute::tuple<cute::C<1>, cute::C<1>>>, 
    cute::tuple<cute::C<0>, cute::C<64>, cute::C<256>, cute::tuple<cute::C<0>, cute::C<0>>>
  >{};

  // Iterator = cute::SM90::GMMA::DescriptorIterator
  auto A = cute::make_tensor(
    descIterator,
    layoutA
  );
  PRINT_EXPR_TYPE("A", A);
  /*
  Tensor<
    cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>, 
    cute::Layout<
      cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>, cute::tuple<cute::C<1>, cute::C<1>>>, 
      cute::tuple<cute::C<0>, cute::C<64>, cute::C<256>, cute::tuple<cute::C<0>, cute::C<0>>>
    >
  >
  */

  // 
  // A.data()
  // 
  PRINT_EXPR_TYPE("A.data() type", A.data());
  // cute::SM90::GMMA::DescriptorIterator &

  auto Ap = A(_,_,_,0);
  PRINT_EXPR_TYPE("Ap = A(_,_,_,0)", Ap);
  auto base0 = Ap.data();
  /*
  cute::Tensor<
    cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>, 
    cute::Layout<
      cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>, 
      cute::tuple<cute::C<0>, cute::C<64>, cute::C<256>>
    >
  >
  */

  for (int k = 0; k < (int)cute::size<2>(A); k++) {
    printf("\n");
    printf("======================================================\n");
    printf("k=%d\n", k);
    printf("======================================================\n");
    auto Ak = Ap(_,_,k);

    auto offset = k * 256;
    auto base_k = base0 + offset;

    PRINT_EXPR_TYPE("Ak = Ap(_,_,k)", Ak);
    for (int m = 0; m < (int)cute::size<1>(A); m++) {
      printf("\nm=%d\n", m);
      auto Am = Ak(_,m);
      PRINT_EXPR_TYPE("Am = Ak(_,m)", Am);
      PRINT_EXPR_TYPE("Am[0]", Am[0]);
      cute::print(Am[0]);
      printf("\n");
    }
  }
}

/*
A:
cute::Tensor<
  cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>, 
  cute::Layout<
    cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>, cute::tuple<cute::C<1>, cute::C<1> > >, 
    cute::tuple<cute::C<0>, cute::C<64>, cute::C<256>, cute::tuple<cute::C<0>, cute::C<0> > > 
  > 
>

Ap = A(_,_,_,0)
cute::Tensor<
  cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>, 
  cute::Layout<
    cute::tuple<cute::C<1>, cute::C<2>, cute::C<4> >, 
    cute::tuple<cute::C<0>, cute::C<64>, cute::C<256> > 
  > 
>

==================
k=0
==================
Ak = Ap(_,_,k):
cute::Tensor<
  cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>, 
  cute::Layout<
    cute::tuple<cute::C<1>, cute::C<2> >, 
    cute::tuple<cute::C<0>, cute::C<64> > 
  > 
>

m=0
Am = Ak(_,m):
cute::Tensor<
  cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>, 
  cute::Layout<
    cute::tuple<cute::C<1> >, 
    cute::tuple<cute::C<0> > 
  > 
>

Am[0]: T = cute::GmmaDescriptor
GmmaDescriptor: 0x4000008000002ce7
  start_addr :  0x2ce7
  leading_off:  0x0000 (0)
  stride_off :  0x0080 (128)
  base_offset:  0x0
  layout_type:  0x1 (B128)


m=1
Am: T = cute::Tensor<cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>, cute::Layout<cute::tuple<cute::C<1> >, cute::tuple<cute::C<0> > > >
Am[0]: T = cute::GmmaDescriptor
GmmaDescriptor: 0x4000008000002d27
  start_addr :  0x2d27
  leading_off:  0x0000 (0)
  stride_off :  0x0080 (128)
  base_offset:  0x0
  layout_type:  0x1 (B128)
*/