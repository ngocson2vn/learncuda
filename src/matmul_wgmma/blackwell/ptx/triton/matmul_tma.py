import torch
from matmul_tma_kernel import matmul_tma

def main():
    a = (torch.rand(256, 1024, dtype=torch.float16).cuda() - 0.5)
    print(f"a.stride: {a.stride()}")
    b = (torch.rand(512, 1024, dtype=torch.float16).cuda() - 0.5)
    print(f"b.stride: {b.stride()}")

    print(f"a: {a}")
    print(f"b: {b}")

    c_torch_matmul = torch.matmul(a, b.T)
    print(f"c_torch_matmul: shape={c_torch_matmul.shape} dtype={c_torch_matmul.dtype} {c_torch_matmul}")

    c_matmul_tma = matmul_tma(a, b, warp_specialize=True)
    print(f"c_matmul_tma: shape={c_matmul_tma.shape} dtype={c_matmul_tma.dtype} {c_matmul_tma}")
    print()

    print("Verify results")
    EPSILON = 1e-2
    matched = True
    mismatch_count = 0
    for i in range(c_torch_matmul.shape[0]):
        for j in range(c_torch_matmul.shape[1]):
            diff = abs((c_torch_matmul[i, j] - c_matmul_tma[i, j]))
            if diff > EPSILON:
                matched = False
                print(f"{c_torch_matmul[i, j]} != {c_matmul_tma[i, j]}")
                mismatch_count += 1

    if matched:
        print(f"OK: matmul_tma matches torch.matmul")
    else:
        print(f"NG: There are {mismatch_count} mismatches between matmul_tma and torch.matmul")

if __name__ == "__main__":
    main()
