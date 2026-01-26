# PTX notes
```MLIR
// %rd7 = a_tensor_map
// %rd8 = b_tensor_map

// %r24 = M

// %r25 = N

// %r1 = tid

// %r2 = warp_id
// %r3 = warp_id

// %p1 = true if warp_id < 8

// %p1 = false if warp_id >= 8

// %r147 = global_smem

// %r214 = TMEM address

// [global_smem+81920] hold TMEM_addr

// %r26 = K

// %r28 = global_smem

// [global_smem+82152] holds 33554689
// 33554689 = 00000010|00000000|00000001|00000001

// %r165 = global_smem + 82048

// %r170 = global_smem + 82096

// %r180 = global_smem + 82144

// %r265 = k_tiles - 1

// %rd18 = the final mbarrier

// %r125 = %r166-69

// Init
//   %r268 = 0
//   %r270 = 0
//   %r271 = 0

// 
// Warp 10
// 
// Init
//   %r267 = 1
```


# Warp 0-7
1. Allocate a TMEM tile of 128 rows by 64 columns (128x64)
2. Reset TMEM tile
3. Init Consumer mbarriers %r165-169
4. Init Producer mbarriers %r170-r174
5. Prime Consumer mbarriers %r165-169
6. Store branch index value `33554689` to `[global_smem+82152]`
7. Arrive at 1st CTA sync point
8. Arrive at 2nd CTA sync point
9. Arrive at 3rd CTA sync point
10. Wait for the final mbarrier %r180 to be satisfied
11. Load matrix D from TMEM to registers
12. Store registers into SMEM in 128B swizzling mode
13. TMA matrix D from SMEM to GMEM
14. Deallocate TMEM
15. Store new branch index value `50529027` to `[global_smem+82152]`
16. Arrive at the 4th CTA sync point
17. Return


# Warp 8-9
## Prologue
1. Arrive at the 1st CTA sync point
2. Arrive at the 2nd CTA sync point

## Enter Copy Loop
1. Wait for Consumer mbarriers %r165-169 are satisfied
2. TMA 64x64xf16 tile_m and tile_n to global_smem
3. Satisfy Producer mbarriers %r170-174

## Epilogue
1. Arrive at the 3rd CTA sync point
2. Jump to `$L__BB0_2`
3. Arrive at the 4th CTA sync point
4. Jump to `$L__BB0_16`
5. Return


# Warp 10
## Prologue
1. Arrive at the 1st CTA sync point
2. Arrive at the 2nd CTA sync point
3. Wait for Producer mbarrier %r170 to be satisfied
4. MMA the first 64x64xf16 tile_m and tile_n
5. Satisfy Consumer mbarrier %r165 (the first 64x64xf16 tile_m and tile_n is consumed)
6. Satisfy the final mbarrier %r180 if k_tile_counter == 0

## Enter MMA Loop
1. Wait for Producer mbarriers %r171-174 to be satisfied.
2. MMA tile_m and tile_n (64x64xf16)
3. Satisfy Consumer mbarriers %r166-169
4. Satisfy the final mbarrier %r180

## Epilogue
1. Arrive the 3rd CTA sync point
2. Jump to `$L__BB0_2`
3. Arrive at the 4th CTA sync point
4. Jump to `$L__BB0_16`
5. Return


# Warp 11:
1. Arrive at 1st CTA sync point
2. Arrive at 2nd CTA sync point
3. Arrive at 3rd CTA sync point
3. Enter `$L__BB0_2`
4. Arrive at the 4th CTA sync point
5. Jump to `$L__BB0_16`
6. Return