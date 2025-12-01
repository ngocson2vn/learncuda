#pragma once

namespace cute {

template <typename T>
void print_host_type(const char* name) {
  const char* func_name = __PRETTY_FUNCTION__;
  // printf("%s\n", func_name);
  char* type = const_cast<char*>(func_name);
  char* ptr = type;
  int n = 0;
  while (*ptr != '=') {
    ptr++;
    n++;
  }

  type = type + (n + 2);
  ptr = type;

  n = 0;
  while (*ptr != ']') {
    ptr++;
    n++;
  }

  printf("%s: %.*s\n", name, n, type);
}

template <typename T>
__device__ void print_device_type(const char* name) {
  const char* func_name = __PRETTY_FUNCTION__;
  char* type = const_cast<char*>(func_name);
  char* ptr = type;
  int n = 0;
  while (*ptr != '=') {
    ptr++;
    n++;
  }

  type = type + (n + 2);
  ptr = type;

  n = 0;
  while (*ptr != ']') {
    ptr++;
    n++;
  }

  *ptr = '\0';

  printf("%s: %s\n", name, type);
}

}

#define PRINT_HOST_EXPR_TYPE(name, expr)                                \
do {                                                                    \
  cute::print_host_type<decltype(expr)>(name);                          \
} while(0)

#define PRINT_DEVICE_EXPR_TYPE(name, expr)                              \
do {                                                                    \
  if (blockIdx.x == 0 && threadIdx.x == 0) {                            \
    cute::print_device_type<decltype(expr)>(name);                      \
  }                                                                     \
} while(0)

#define PRINT_HOST_FULL_TYPE(name, type)                                \
do {                                                                    \
  cute::print_host_type<type>(name);                                    \
} while(0)

#define PRINT_DEVICE_FULL_TYPE(name, type)                              \
do {                                                                    \
  if (blockIdx.x == 0 && threadIdx.x == 0) {                            \
    cute::print_device_type<type>(name);                                \
  }                                                                     \
} while(0)

#define PRINT_HOST_INT_VALUE(name, val)                                 \
do {                                                                    \
  printf("%s: %d\n", name, val);                                        \
} while(0)

#define PRINT_DEVICE_INT_VALUE(name, val)                               \
do {                                                                    \
  if (blockIdx.x == 0 && threadIdx.x == 0) {                            \
    printf("%s: %d\n", name, val);                                      \
  }                                                                     \
} while(0)

#define PRINT_DEVICE_FUNC_SIG()                                         \
do {                                                                    \
  if (blockIdx.x == 0 && threadIdx.x == 0) {                            \
    printf("\n%s:%d: %s\n", __FILE__, __LINE__, __PRETTY_FUNCTION__);   \
  }                                                                     \
} while(0)

#define PRINT_HOST_FUNC_SIG()                                           \
do {                                                                    \
  printf("\n%s:%d: %s\n", __FILE__, __LINE__, __PRETTY_FUNCTION__);     \
} while(0)

#define CUTE_DEVICE_PRINT(obj)                                          \
do {                                                                    \
  if (threadIdx.x == 0) {                                               \
    cute::print(obj);                                                   \
  }                                                                     \
} while(0)


#if defined(__CUDA_ARCH__)
#define PRINT_FUNC_SIG() PRINT_DEVICE_FUNC_SIG()
#define PRINT_EXPR_TYPE(name, expr) PRINT_DEVICE_EXPR_TYPE(name, expr)
#define CUTE_PRINT(x) CUTE_DEVICE_PRINT(x)
#define PRINT_FULL_TYPE(name, type) PRINT_DEVICE_FULL_TYPE(name, type)
#define PRINT_INT_VALUE(name, val) PRINT_DEVICE_INT_VALUE(name, val)
#else
#define PRINT_FUNC_SIG() PRINT_HOST_FUNC_SIG()
#define PRINT_EXPR_TYPE(name, expr) PRINT_HOST_EXPR_TYPE(name, expr)
#define CUTE_PRINT(x)
#define PRINT_FULL_TYPE(name, type) PRINT_HOST_FULL_TYPE(name, type)
#define PRINT_INT_VALUE(name, val) PRINT_HOST_INT_VALUE(name, val)
#endif
