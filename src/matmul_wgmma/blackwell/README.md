# GEMM Kernel Design with TMEM, TMA, and MMA v5
## Warp 0-7
### Prologue
1. Allocate a TMEM tile of 128 rows by 64 columns (128x64)
2. Reset TMEM tile
3. Init Consumer mbarriers
4. Init Producer mbarriers
5. Prime Consumer mbarriers

### Enter Wait State
6. Arrive at the 1st CTA sync point
7. Arrive at the 2nd CTA sync point
8. Wait for the final mbarrier to be satisfied

### Epilogue
9. Load matrix D from TMEM to registers
10. Store registers into SMEM in 128B swizzling mode
11. TMA matrix D from SMEM to GMEM
12. Deallocate TMEM


## Warp 8
### Prologue
1. Arrive at the 1st CTA sync point

### Enter Copy Loop
2. Wait for Consumer mbarriers to be satisfied
3. TMA 64x64xf16 tile_m and tile_n to global_smem
4. Satisfy Producer mbarriers

### Epilogue
5. Arrive at the 2nd CTA sync point


## Warp 9
### Prologue
1. Arrive at the 1st CTA sync point
2. Wait for the 1st Producer mbarrier to be satisfied
3. MMA the 1st 64x64xf16 tile_m and tile_n
4. Satisfy the 1st Consumer mbarrier (the 1st 64x64xf16 tile_m and tile_n is consumed)
5. Satisfy the final mbarrier if k_tile_counter == 0

### Enter MMA Loop
6. Wait for Producer mbarriers to be satisfied.
7. MMA tile_m and tile_n (64x64xf16)
8. Satisfy Consumer mbarriers
9. Satisfy the final mbarrier if k_tile_counter == 1

### Epilogue
10. Arrive at the 2nd CTA sync point


## Warp 0-9
1. Arrive at the 3rd CTA sync point
2. Return

<br/>


# Build and Run
**Prerequisite**: CUDA Toolkit >= 12.8
```Bash
# Verify mode
make
./mmav5_with_tma 512 512 16

# Debug mode
make debug
./mmav5_with_tma_debug

# Profile mode
make profile
./mmav5_with_tma_profile 512 512 16
```
