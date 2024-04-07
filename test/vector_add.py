import torch
import argparse

from cpp.gorby_vector_add import vector_add_1d_inplace

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--length", type=int, default=4)
    parser.add_argument("-d", "--device", type=str, default="cuda")
    args = parser.parse_args()

    pass
