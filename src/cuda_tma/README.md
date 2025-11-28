# NOTE
The latest TMA kernel is [main3.cu](./main3.cu).
```
TMA kernel launch: 4 blocks of 32 threads.
Each block operates on a quarter of matrix A along column dimension:
- Copy a quarter of matrix A to shared memory
- Update each element of the quarter residing in shared memory
- Transfer the updated quarter from shared memory back to global memory
|----------------------------|
|     quarter1: 16x32        |
|----------------------------|
|     quarter2: 16x32        |
|----------------------------|
|     quarter3: 16x32        |
|----------------------------|
|     quarter4: 16x32        |
|----------------------------|
```

## Prerequisites
```Bash
GPU: NVIDIA H100
Driver Version: >=535.129.03
CUDA Version: 12.4
```

## Build
```Bash
make
```

## Run
```Bash
./main3
```
Matrix A is dumped to `output.txt`.


# NOTE about mbarrier
```C++
  if (tid == 0) {
    // mbarrier.init.shared::cta [addr], expected_tx;
    mbarrier_init(&mbar, 1);

    int expected_bytes = SMEM_ROWS * SMEM_COLS * sizeof(DataType);
    mbarrier_arrive_and_expect_tx_bytes(&mbar, expected_bytes);

    // Initiate bulk tensor copy.
    cp_async_bulk_tensor_2d_global_to_shared(
      smem_buffer,
      &tensor_map_input,
      col,
      row,
      &mbar
    );

    printf("[tma_kernel] threadIdx.x %d initiated bulk tensor copy\n", threadIdx.x);
  }

  // Syncthreads to ensure that initialized barrier is visible to all threads.
  __syncthreads();

  // Wait for the data to transfer from global -> shared
  // NOTE: pay attention to phase parity
  mbarrier_wait(&mbar, 1);
  printf("[tma_kernel] threadIdx.x %d arrived\n", threadIdx.x);
```
The code hangs when using `mbarrier_wait(&mbar, 1)` because the mbarrier completion triggered by the TMA transfer (i.e., when all transaction bytes have arrived and `pending_tx == 0`) occurs when the phase parity is **0**, not 1.

Here is the exact sequence of events that explains why parity **0** works and parity **1** deadlocks:

Initial state after `mbarrier.init(&mbar, 1)`:  
- expected arrivals = 1  
- arrival count = 0  
- phase parity = 0 (this is guaranteed by the PTX specification: `mbarrier.init` always initializes the phase to zero)  
- pending_tx = 0 (or undefined, but irrelevant yet)

Thread 0 executes `mbarrier.arrive.expect_tx` (bytes = SMEM_ROWS × SMEM_COLS × sizeof(DataType)):  
- Increments arrival count 0 → 1  
- This satisfies expected arrivals (1 == 1) → arrival count is reset to 0 and phase is toggled → **phase parity = 1**  
- Because this was the completing arrive and the instruction has `.expect_tx`, pending_tx is now set to the byte count  
State now: phase = 1, arrival count = 0, pending_tx = large positive number

At this point the TMA transfer is issued (`cp.async.bulk.tensor.2d...mbarrier::complete_tx::bytes`).  
The hardware now starts issuing `complete_tx` operations that decrement `pending_tx`.  
When the last `complete_tx` makes `pending_tx <= 0`:  
- Hardware sets `pending_tx = 0`  
- Hardware performs an implicit arrive: arrival count 0 → 1 → satisfies expected → arrival count reset to 0, phase toggled again → **phase parity = 0**

The `mbarrier.try_wait.parity` instruction (inside your `mbarrier_wait`) succeeds when **both** of these are true:  
1. Current phase parity == the requested parity  
2. (arrival count >= expected arrivals) **OR** (pending_tx == 0)

After the explicit `arrive.expect_tx` (phase = 1):  
- arrival count = 0 < 1  → arrivals not complete  
 pending_tx > 0           → tx not complete  
→ try_wait on parity 1 returns false (even though phase == 1, the completion condition is not satisfied)

After the implicit arrive triggered by `pending_tx == 0` (phase = 0):  
- pending_tx == 0 → the OR condition is true  
→ try_wait on parity 0 now returns true

Therefore the only phase at which the completion condition is satisfied is parity **0**.  
Waiting on parity 1 can never succeed because when phase == 1, `pending_tx` is still > 0 and arrival count = 0 < 1, so neither side of the OR is true. The barrier will never appear "complete" at parity 1.

This is why your kernel works with `mbarrier_wait(&mbar, 0)` and deadlocks with `mbarrier_wait(&mbar, 1)`.

This pattern (init with count=1 + single `arrive.expect_tx`) always requires waiting on parity **0** on Hopper/SM_90+.  
If you want to wait on parity 1 instead, you would need a different initialization pattern (e.g., initialize with expected arrivals = 2, have thread 0 do a plain `mbarrier.arrive` (no expect_tx), then a second thread or the same thread do `mbarrier.arrive.expect_tx`, or capture the phase bit returned by some variants of the arrive instruction). But the simplest and most common fix is just to keep the wait at 0 — it is correct as-is.

So the solid reason is: **the transaction-completion-triggered implicit arrive toggles the phase back to 0, and that is the only phase where `pending_tx == 0` (the condition required for try_wait to succeed) holds**. Parity 1 is the intermediate phase where the barrier is not yet complete from the transaction perspective.