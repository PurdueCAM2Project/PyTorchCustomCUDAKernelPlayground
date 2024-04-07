import torch
import argparse

from cpp.gorby_nvtx import nvtx_init
from cpp.gorby_vector_add import vector_add_1d_inplace

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--length", type=int, default=4)
    parser.add_argument("-d", "--device", type=str, default="cuda")
    args = parser.parse_args()

    ### For fun - start nvtx!
    nvtx_init()

    print("Created NVTX context!")

    x = torch.ones(size=(args.length,), dtype=torch.float32, device=args.device)
    y = torch.ones(size=(args.length,), dtype=torch.float32, device=args.device)

    if args.length >= 4:
        print(f"x pre: {x[0:2]}...{x[-2:None]}\ny: {y[0:2]}...{y[-2:None]}")
    else:
        print(f"x pre: {x}\ny: {y}")

    vector_add_1d_inplace(x, y)

    if args.length >= 4:
        print(f"x post: {x[0:2]}...{x[-2:None]}\ny: {y[0:2]}...{y[-2:None]}")
    else:
        print(f"x post: {x}\ny: {y}")
