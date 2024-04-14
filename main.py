import torch
import argparse
import torch.utils.benchmark as bench
from typing import Any, Literal, List, Dict

### Local Ops!
from ops.gorby_nvtx import nvtx_init
from ops.gorby_vector_add import vector_add_1d_inplace
from ops.gorby_sdpa import sdpa_forward

### Create a benchmark function (very simple)
def benchmark_milliseconds(model: torch.nn.Module) -> float:
    ### Do the benchmark!
    t0 = bench.Timer(
        stmt=f"model()",
        globals={"model": model},
        num_threads=1,
    )

    return t0.blocked_autorange(min_run_time=4.0).median * 1e6

###
### Dummy SDPA / GEMM Model
###
class GorbyGEMM(torch.nn.Module):
    def __init__(self, M : int = 128, N : int = 128, K : int = 128, device : torch.device = torch.device("cuda:0")) -> None:
        super().__init__()
        self.M = M
        self.N = N
        self.K = K
        self.A = torch.ones(size=(M,N), device=device)
        self.B = torch.ones(size=(N,K), device=device)
        self.C = torch.zeros(size=(M,K), device=device)

    def forward(self) -> torch.Tensor:
        return sdpa_forward(self.A, self.B, self.C)
    
###
### Native Torch GEMM
###
class NativeGEMM(torch.nn.Module):
    def __init__(self, M : int = 128, N : int = 128, K : int = 128, device : torch.device = torch.device("cuda:0")) -> None:
        super().__init__()
        self.M = M
        self.N = N
        self.K = K
        self.A = torch.ones(size=(M,N), device=device)
        self.B = torch.ones(size=(N,K), device=device)
        self.C = torch.zeros(size=(M,K), device=device)

    def forward(self) -> torch.Tensor:
        return self.A @ self.B + self.C

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=128)
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("-d", "--device", type=str, default="cuda")
    args = parser.parse_args()

    ### Create Device
    device = torch.device(args.device)

    ### For fun - start nvtx!
    # nvtx_init()
    # print("Created NVTX context!")
    
	###
	### SDPA / Matmul Testing
	###
    gorby_gemm_module = GorbyGEMM(args.M, args.N, args.K, device=device)
    native_gemm_module = NativeGEMM(args.M, args.N, args.K, device=device)

    ### Forward
    D_gorby_gemm = gorby_gemm_module()
    D_native_gemm = native_gemm_module()

    ### Info
    print(f"native / gorby gemm D shapes: {D_native_gemm.shape} / {D_gorby_gemm.shape}")
    print(f"native gemm D: {D_native_gemm}")
    print(f"gorby_gemm D: {D_gorby_gemm}")

    ### Benchmarking
    native_ms = benchmark_milliseconds(native_gemm_module)
    gorby_ms = benchmark_milliseconds(gorby_gemm_module)
    print(f"native / gorby median ms: {native_ms:.2f} / {gorby_ms:.2f}")

    percent_diff = (native_ms-gorby_ms) / native_ms * 100.0
    perecent_diff_string = "higher" if percent_diff < 0 else "lower"
    print(f"gorby latency is {abs(percent_diff):.2f}% {perecent_diff_string}")