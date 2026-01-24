#include <mutex>
#include <thread>
#include <type_traits>
#include <thrust/host_vector.h>

// mat_A is M-major (row-major: K is contiguous)
// mat_B is N-major (row-major: K is contiguous)
template <typename ABType, typename DType>
void matmul_cpu(
  const thrust::host_vector<ABType>& mat_A,
  const thrust::host_vector<ABType>& mat_B,
  thrust::host_vector<DType>& mat_D, 
  int M, int N, int K
) {
  if (std::is_same<ABType, DType>()) {
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        int idx = m + n * M;
        for (int k = 0; k < K; k++) {
          mat_D[idx] += mat_A[m + k * M] * mat_B[n + k * N];
        }
      }
    }
  } else {
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        int idx = m + n * M;
        for (int k = 0; k < K; k++) {
          auto mat_A_mk = DType(mat_A[m + k * M]);
          auto mat_B_nk = DType(mat_B[n + k * N]);
          mat_D[idx] += mat_A_mk * mat_B_nk;
        }
      }
    }
  }
}

template <typename ABType, typename DType, int M_TILE, int N_TILE>
void matmul_cpu_tile(
  const ABType* mat_A,
  const ABType* mat_B,
  DType* mat_D,
  const int stride_A[2], const int stride_B[2], const int stride_D[2], 
  const int M, const int N, const int K,
  const int m_start, const int n_start
) {
  std::vector<std::thread> threads;
  for (int i = 0; i < M_TILE; i++) {
    int m = m_start + i;
    for (int j = 0; j < N_TILE; j++) {
      int n = n_start + j;
      int idx = m * stride_D[0] + n * stride_D[1];
      for (int k = 0; k < K; k++) {
        auto mat_A_mk = static_cast<DType>(mat_A[m * stride_A[0] + k * stride_A[1]]);
        auto mat_B_nk = static_cast<DType>(mat_B[n * stride_B[0] + k * stride_B[1]]);
        mat_D[idx] += mat_A_mk * mat_B_nk;
      }
    }
  }
}

template <typename ABType, typename DType, int M_TILE, int N_TILE, int MAX_THREAD_COUNT = 1024>
void matmul_cpu_parallel(
  const thrust::host_vector<ABType>& mat_A,
  const thrust::host_vector<ABType>& mat_B,
  thrust::host_vector<DType>& mat_D,
  const int stride_A[2], const int stride_B[2], const int stride_D[2],
  const int M, const int N, const int K
) {
  const int M_TILE_COUNT = M / M_TILE;
  const int N_TILE_COUNT = N / N_TILE;
  const int total_tiles = M_TILE_COUNT * N_TILE_COUNT;
  const int total_batches = total_tiles / MAX_THREAD_COUNT + ((total_tiles % MAX_THREAD_COUNT) > 0 ? 1 : 0);
  printf("Total batches of tiles to be processed: %d\n", total_batches);

  std::vector<std::thread> threads;
  int batch_count = 0;
  for (int m_tile = 0; m_tile < M_TILE_COUNT; m_tile++) {
    for (int n_tile = 0; n_tile < N_TILE_COUNT; n_tile++) {
      threads.emplace_back(
        matmul_cpu_tile<ABType, DType, M_TILE, N_TILE>, mat_A.data(), mat_B.data(), mat_D.data(), stride_A, stride_B, stride_D, M, N, K, m_tile * M_TILE, n_tile * N_TILE
      );

      if (threads.size() == MAX_THREAD_COUNT) {
        for (auto& t : threads) {
          t.join();
        }
        batch_count++;
        printf("Done processing batch %d of %d tiles\n", batch_count, threads.size());
        threads.clear();
      }
    }
  }

  if (threads.size() > 0) {
    for (auto& t : threads) {
      t.join();
    }
    batch_count++;
    printf("Done processing batch %d of %d tiles\n", batch_count, threads.size());
  }
}

static std::atomic<int> g_ok(0);
static std::atomic<int> g_ng(0);

template <typename DType>
void verify_worker(const DType* cpu_data, const DType* gpu_data, int start_idx, int numElements) {
  int ok = 0;
  int ng = 0;
  constexpr DType EPSILON = 1.0e-3;
  for (int i = 0; i < numElements; i++) {
    int idx = start_idx + i;
    DType diff = std::abs(cpu_data[idx] - gpu_data[idx]);
    if (diff < EPSILON) {
      ok++;
    } else {
      ng++;
    }
  }

  g_ok.fetch_add(ok);
  if (ng > 0) {
    g_ng.fetch_add(ng);
  }
}


template <typename DType, int MAX_THREAD_COUNT = 1024>
void verify(const thrust::host_vector<DType>& h_D_cpu, const thrust::host_vector<DType>& h_D_gpu, int M, int N) {
  // Reset global counters
  g_ok = 0; g_ng = 0;
  std::vector<std::thread> verifyThreads;
  const int kNumElements = (M * N) / MAX_THREAD_COUNT;

  printf("\nVerifying results\n");
  for (int i = 0; i < MAX_THREAD_COUNT; i++) {
    int idx_start = i * kNumElements;
    verifyThreads.emplace_back(verify_worker<DType>, h_D_cpu.data(), h_D_gpu.data(), idx_start, kNumElements);
  }

  for (auto& t : verifyThreads) {
    t.join();
  }
  printf("OK=%d NG=%d\n\n", g_ok.fetch_add(0), g_ng.fetch_add(0));
}
