# Refs
Swizzles and their usage in CuTeDSL Kernels<br/>
https://veitner.bearblog.dev/swizzles-and-their-usage-in-cutedsl-kernels/

# Swizzle<3,4,3>
`Swizzle<3,4,3>` is a specific instantiation of the Swizzle template class in NVIDIA's CuTe (part of the CUTLASS library), designed to perform bit-level address remapping (swizzling) on offsets used for accessing GPU shared memory. It is commonly referred to as the "128-byte swizzling function" because its permutation pattern repeats every 128 bytes (1024 bits), making it suitable for optimizing memory access in tiles or buffers of that granularity or multiples thereof. This swizzle is particularly used in high-performance kernels like GEMM (General Matrix Multiply) and Flash Attention to reorganize data layouts, ensuring efficient vectorized loads/stores while mitigating performance bottlenecks.

### How Swizzle<3,4,3> Works
The Swizzle template is parameterized by three integer constants:
- `BBits` (first parameter, 3): The width of the bit fields being manipulated (a 3-bit mask).
- `MBase` (second parameter, 4): The number of least-significant bits (LSBs) that remain unchanged, preserving alignment for vectorized accesses (e.g., 2^4 = 16 bytes, often aligning with 128-bit vectors like `uint128_t` for 16 INT8 elements).
- `SShift` (third parameter, 3): The shift amount, which determines the relative positioning of the bit fields and the direction of the shift (positive here, indicating the source bits are higher than the target bits).

The core operation is a functor that takes an input offset (a logical address in shared memory) and computes a swizzled (physical) offset via a bitwise XOR:
```C++
swizzled_offset = offset ^ ( (offset & yyy_msk) >> SShift )
```
This is efficient as it's a simple bit manipulation, computable at compile-time or runtime with minimal overhead.

#### Bit Masks and Fields
The masks are precomputed constants:
- `bit_msk = (1 << 3) - 1 = 0b111` (7 in decimal): Base mask for a 3-bit field.
- `yyy_msk = bit_msk << (MBase + SShift) = 0b111 << (4 + 3) = 0b111 << 7 = 0b1110000000` (binary, affecting bits 7–9, zero-indexed from LSB as bit 0).
- `zzz_msk = bit_msk << MBase = 0b111 << 4 = 0b1110000` (affecting bits 4–6).
- `msk_sft = 3`: The right-shift amount (since SShift > 0).

The "YYY" field (source bits) is at positions 7–9 (higher bits), and the "ZZZ" field (target bits) is at 4–6 (lower bits). <br/>
These fields are disjoint (separated by |SShift| = 3 bits, meeting the requirement |SShift| >= BBits to avoid overlap).

#### Bit Manipulation Example
Consider an offset like `0b...00001111111111` (1023 in decimal, binary shown for relevant lower bits):
1. Mask the YYY bits: `offset & yyy_msk = 0b...00001110000000` (bits 7–9 extracted as 0b111).
2. Shift right by 3: `0b...00000001110000` (now aligned to bits 4–6 as 0b111).
3. XOR with original: `0b...00001111111111 ^ 0b...00000001110000 = 0b...00001110001111` (bits 4–6 flipped from 0b111 to 0b000 via XOR).

**Effectively:** <-- This is the key point
- Bit 7 flips bit 4 (if bit 7 is 1).
- Bit 8 flips bit 5 (if bit 8 is 1).
- Bit 9 flips bit 6 (if bit 9 is 1).
- Bits 0–3 remain unchanged (due to MBase=4).
- Higher bits (10+) are unaffected because of the nature of XOR

### Why It Avoids Bank Conflicts
GPU shared memory (SMEM) on NVIDIA architectures (e.g., Ampere, Hopper) is divided into 32 banks, each 32 bits (4 bytes) wide. A warp (32 threads) can access SMEM in one cycle if each thread hits a unique bank—otherwise, bank conflicts occur, serializing accesses and reducing throughput (e.g., a 32-way conflict degrades performance by up to 32×).

Conflicts arise in patterns like:
- **Strided accesses**: E.g., reading a column in a row-major 2D matrix (stride = row size). All threads might compute offsets like `base + threadIdx.x * stride`, mapping to the same bank if the stride aligns poorly (e.g., multiple of 128 bytes).
- **Vectorized loads**: Common in GEMM for INT8/FP16 data, where threads load 128-bit vectors (16 bytes), but without remapping, column accesses cluster in few banks.

`Swizzle<3,4,3>` mitigates this by scrambling (make something jumbled or disordered) the address bits that determine the bank index by scrambling the **logical byte offset** (because the physical address = base + **logical byte offset**):
- Bank ID = `(addr >> 2) % 32` = bits[6:2] of the logical byte offset (since >>2 ignores intra-word bits 0–1, and %32 uses 5 bits for 32 banks).
- The swizzle flips bits `4–6` (which overlap the bank selector bits `2–6`) based on higher bits `7–9`. This spreads accesses across banks:
  - For a warp accessing a "column" (strided by 128 bytes = 2^7), the higher bits (7–9) vary per "row," providing unique XOR constants `c`.
  - Result: Bank indices become permuted (e.g., 0 to 31 uniquely), ensuring no two threads hit the same bank.
- Preserves contiguity for rows (low bits unchanged) while making columns/diagonals conflict-free.
- In TMA (Tensor Memory Accelerator on Hopper+), swizzling happens during async GMEM-to-SMEM copies, pre-arranging data for optimal warp accesses.

#### Example in Practice
Given matrix A[8x64xb16] with row-major layout in global memory as follows:
```
0.0   1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0  10.0  11.0  12.0  13.0  14.0  15.0  16.0  17.0  18.0  19.0  20.0  21.0  22.0  23.0  24.0  25.0  26.0  27.0  28.0  29.0  30.0  31.0  32.0  33.0  34.0  35.0  36.0  37.0  38.0  39.0  40.0  41.0  42.0  43.0  44.0  45.0  46.0  47.0  48.0  49.0  50.0  51.0  52.0  53.0  54.0  55.0  56.0  57.0  58.0  59.0  60.0  61.0  62.0  63.0 
0.0   1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0  10.0  11.0  12.0  13.0  14.0  15.0  16.0  17.0  18.0  19.0  20.0  21.0  22.0  23.0  24.0  25.0  26.0  27.0  28.0  29.0  30.0  31.0  32.0  33.0  34.0  35.0  36.0  37.0  38.0  39.0  40.0  41.0  42.0  43.0  44.0  45.0  46.0  47.0  48.0  49.0  50.0  51.0  52.0  53.0  54.0  55.0  56.0  57.0  58.0  59.0  60.0  61.0  62.0  63.0 
0.0   1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0  10.0  11.0  12.0  13.0  14.0  15.0  16.0  17.0  18.0  19.0  20.0  21.0  22.0  23.0  24.0  25.0  26.0  27.0  28.0  29.0  30.0  31.0  32.0  33.0  34.0  35.0  36.0  37.0  38.0  39.0  40.0  41.0  42.0  43.0  44.0  45.0  46.0  47.0  48.0  49.0  50.0  51.0  52.0  53.0  54.0  55.0  56.0  57.0  58.0  59.0  60.0  61.0  62.0  63.0 
0.0   1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0  10.0  11.0  12.0  13.0  14.0  15.0  16.0  17.0  18.0  19.0  20.0  21.0  22.0  23.0  24.0  25.0  26.0  27.0  28.0  29.0  30.0  31.0  32.0  33.0  34.0  35.0  36.0  37.0  38.0  39.0  40.0  41.0  42.0  43.0  44.0  45.0  46.0  47.0  48.0  49.0  50.0  51.0  52.0  53.0  54.0  55.0  56.0  57.0  58.0  59.0  60.0  61.0  62.0  63.0 
0.0   1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0  10.0  11.0  12.0  13.0  14.0  15.0  16.0  17.0  18.0  19.0  20.0  21.0  22.0  23.0  24.0  25.0  26.0  27.0  28.0  29.0  30.0  31.0  32.0  33.0  34.0  35.0  36.0  37.0  38.0  39.0  40.0  41.0  42.0  43.0  44.0  45.0  46.0  47.0  48.0  49.0  50.0  51.0  52.0  53.0  54.0  55.0  56.0  57.0  58.0  59.0  60.0  61.0  62.0  63.0 
0.0   1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0  10.0  11.0  12.0  13.0  14.0  15.0  16.0  17.0  18.0  19.0  20.0  21.0  22.0  23.0  24.0  25.0  26.0  27.0  28.0  29.0  30.0  31.0  32.0  33.0  34.0  35.0  36.0  37.0  38.0  39.0  40.0  41.0  42.0  43.0  44.0  45.0  46.0  47.0  48.0  49.0  50.0  51.0  52.0  53.0  54.0  55.0  56.0  57.0  58.0  59.0  60.0  61.0  62.0  63.0 
0.0   1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0  10.0  11.0  12.0  13.0  14.0  15.0  16.0  17.0  18.0  19.0  20.0  21.0  22.0  23.0  24.0  25.0  26.0  27.0  28.0  29.0  30.0  31.0  32.0  33.0  34.0  35.0  36.0  37.0  38.0  39.0  40.0  41.0  42.0  43.0  44.0  45.0  46.0  47.0  48.0  49.0  50.0  51.0  52.0  53.0  54.0  55.0  56.0  57.0  58.0  59.0  60.0  61.0  62.0  63.0 
0.0   1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0  10.0  11.0  12.0  13.0  14.0  15.0  16.0  17.0  18.0  19.0  20.0  21.0  22.0  23.0  24.0  25.0  26.0  27.0  28.0  29.0  30.0  31.0  32.0  33.0  34.0  35.0  36.0  37.0  38.0  39.0  40.0  41.0  42.0  43.0  44.0  45.0  46.0  47.0  48.0  49.0  50.0  51.0  52.0  53.0  54.0  55.0  56.0  57.0  58.0  59.0  60.0  61.0  62.0  63.0 
```
Now we want to copy this matrix A to shared memory and store it in Swizzle<3,4,3> mode. We need to perform the following operations:
```
Row 0:
  - logical_byte_offset = j*2, j = 0, ..., 63
  - swizzled_byte_offset = logical_byte_offset because bits 7-9 are not set, bits 4-6 are not flipped

Row 1:
  - logical_byte_offset = 64*2 + j*2 = 2^7 + j*2, j = 0, ..., 63 (Row 0 occupies 64*2 bytes)
  - swizzled_byte_offset = (2^7 + j*2) xor 2^4 because bit 7 is set
    - Columns 0 ~ 7:
      - swizzled_byte_offset = 2^7 + 2^4 + k*2, k = 0, ..., 7
    - Columns 8 ~ 15:
      - swizzled_byte_offset = 2^7 + k*2, k = 0, ..., 7
    - Columns 16 ~ 23:
      - swizzled_byte_offset = (2^7 + (2^4 + k)*2) xor 2^4 = (2^7 + 2^5 + k*2) xor 2^4 = 2^7 + 2^5 + 2^4 + k*2
    - Columns 24 ~ 31:
      - swizzled_byte_offset = (2^7 + (2^4 + 2^3 + k)*2) xor 2^4 = (2^7 + 2^5 + 2^4 + k*2) xor 2^4 = 2^7 + 2^5 + k*2
    - Columns 32 ~ 39:
      - swizzled_byte_offset = (2^7 + (2^5 + k)*2) xor 2^4 = (2^7 + 2^6 + k*2) xor 2^4 = 2^7 + 2^6 + 2^4 + k*2
    - Columns 40 ~ 47:
      - swizzled_byte_offset = (2^7 + (2^5 + 2^3 + k)*2) xor 2^4 = (2^7 + 2^6 + 2^4 + k*2) xor 2^4 = 2^7 + 2^6 + k*2
    - Columns 48 ~ 55:
      - swizzled_byte_offset = (2^7 + (2^5 + 2^4 + k)*2) xor 2^4 = (2^7 + 2^6 + 2^5 + k*2) xor 2^4 = 2^7 + 2^6 + 2^5 + 2^4 + k*2
    - Columns 56 ~ 63:
      - swizzled_byte_offset = (2^7 + (2^5 + 2^4 + 2^3 + k)*2) xor 2^4 = (2^7 + 2^6 + 2^5 + 2^4 + k*2) xor 2^4 = 2^7 + 2^6 + 2^5 + k*2
Row 2:
  - logical_byte_offset = 2*64*2 + j*2 = 2^8 + j*2, j = 0, ..., 63 (Row 0-1 occupy 2*64*2 bytes)
  - swizzled_byte_offset = (2^8 + j*2) xor 2^5 because bit 8 is set
...
Row 7:
  - logical_byte_offset = 7*64*2 + j*2 = 2^9 + 2^8 + 2^7 + j*2, j = 0, ..., 63 (Row 0-6 occupy 7*64*2 bytes)
  - swizzled_byte_offset = (2^9 + 2^8 + 2^7 + j*2) xor (2^6 + 2^5 + 2^4) because bit 6-4 are set
```

The matrix A in shared memory:<br/>
```
 0.0   1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0  10.0  11.0  12.0  13.0  14.0  15.0  16.0  17.0  18.0  19.0  20.0  21.0  22.0  23.0  24.0  25.0  26.0  27.0  28.0  29.0  30.0  31.0  32.0  33.0  34.0  35.0  36.0  37.0  38.0  39.0  40.0  41.0  42.0  43.0  44.0  45.0  46.0  47.0  48.0  49.0  50.0  51.0  52.0  53.0  54.0  55.0  56.0  57.0  58.0  59.0  60.0  61.0  62.0  63.0
 8.0   9.0  10.0  11.0  12.0  13.0  14.0  15.0   0.0   1.0   2.0   3.0   4.0   5.0   6.0   7.0  24.0  25.0  26.0  27.0  28.0  29.0  30.0  31.0  16.0  17.0  18.0  19.0  20.0  21.0  22.0  23.0  40.0  41.0  42.0  43.0  44.0  45.0  46.0  47.0  32.0  33.0  34.0  35.0  36.0  37.0  38.0  39.0  56.0  57.0  58.0  59.0  60.0  61.0  62.0  63.0  48.0  49.0  50.0  51.0  52.0  53.0  54.0  55.0
16.0  17.0  18.0  19.0  20.0  21.0  22.0  23.0  24.0  25.0  26.0  27.0  28.0  29.0  30.0  31.0   0.0   1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0  10.0  11.0  12.0  13.0  14.0  15.0  48.0  49.0  50.0  51.0  52.0  53.0  54.0  55.0  56.0  57.0  58.0  59.0  60.0  61.0  62.0  63.0  32.0  33.0  34.0  35.0  36.0  37.0  38.0  39.0  40.0  41.0  42.0  43.0  44.0  45.0  46.0  47.0
24.0  25.0  26.0  27.0  28.0  29.0  30.0  31.0  16.0  17.0  18.0  19.0  20.0  21.0  22.0  23.0   8.0   9.0  10.0  11.0  12.0  13.0  14.0  15.0   0.0   1.0   2.0   3.0   4.0   5.0   6.0   7.0  56.0  57.0  58.0  59.0  60.0  61.0  62.0  63.0  48.0  49.0  50.0  51.0  52.0  53.0  54.0  55.0  40.0  41.0  42.0  43.0  44.0  45.0  46.0  47.0  32.0  33.0  34.0  35.0  36.0  37.0  38.0  39.0
32.0  33.0  34.0  35.0  36.0  37.0  38.0  39.0  40.0  41.0  42.0  43.0  44.0  45.0  46.0  47.0  48.0  49.0  50.0  51.0  52.0  53.0  54.0  55.0  56.0  57.0  58.0  59.0  60.0  61.0  62.0  63.0   0.0   1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0  10.0  11.0  12.0  13.0  14.0  15.0  16.0  17.0  18.0  19.0  20.0  21.0  22.0  23.0  24.0  25.0  26.0  27.0  28.0  29.0  30.0  31.0
40.0  41.0  42.0  43.0  44.0  45.0  46.0  47.0  32.0  33.0  34.0  35.0  36.0  37.0  38.0  39.0  56.0  57.0  58.0  59.0  60.0  61.0  62.0  63.0  48.0  49.0  50.0  51.0  52.0  53.0  54.0  55.0   8.0   9.0  10.0  11.0  12.0  13.0  14.0  15.0   0.0   1.0   2.0   3.0   4.0   5.0   6.0   7.0  24.0  25.0  26.0  27.0  28.0  29.0  30.0  31.0  16.0  17.0  18.0  19.0  20.0  21.0  22.0  23.0
48.0  49.0  50.0  51.0  52.0  53.0  54.0  55.0  56.0  57.0  58.0  59.0  60.0  61.0  62.0  63.0  32.0  33.0  34.0  35.0  36.0  37.0  38.0  39.0  40.0  41.0  42.0  43.0  44.0  45.0  46.0  47.0  16.0  17.0  18.0  19.0  20.0  21.0  22.0  23.0  24.0  25.0  26.0  27.0  28.0  29.0  30.0  31.0   0.0   1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0  10.0  11.0  12.0  13.0  14.0  15.0
56.0  57.0  58.0  59.0  60.0  61.0  62.0  63.0  48.0  49.0  50.0  51.0  52.0  53.0  54.0  55.0  40.0  41.0  42.0  43.0  44.0  45.0  46.0  47.0  32.0  33.0  34.0  35.0  36.0  37.0  38.0  39.0  24.0  25.0  26.0  27.0  28.0  29.0  30.0  31.0  16.0  17.0  18.0  19.0  20.0  21.0  22.0  23.0   8.0   9.0  10.0  11.0  12.0  13.0  14.0  15.0   0.0   1.0   2.0   3.0   4.0   5.0   6.0   7.0
```
