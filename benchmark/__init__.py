import torch
import torch.utils.benchmark as bench
from typing import Any, Callable, Tuple, Set, List

###
### Benchmark milliseconds (optional NVTX markers?)
###
def benchmark_milliseconds(
    f : Callable,
    *args,
    **kwargs,
    ) -> bench.Measurement:
    ### Set a minimum runtime 
    MIN_RUNTIME = 32.0

    t0 = bench.Timer(
        stmt=f"f(*args,**kwargs)",
        globals={
            "f" : f,
            "args" : args,
            "kwargs" : kwargs,
        }
    )

    return t0.blocked_autorange(min_run_time=MIN_RUNTIME)


###
### Benchmark microseconds (optional NVTX markers?)
###
def benchmark_microseconds(
    f : Callable,
    *args,
    **kwargs
    ) -> bench.Measurement:

    ### Set a minimum runtime 
    MIN_RUNTIME = 32.0

    t0 = bench.Timer(
        stmt=f"f(*args,**kwargs)",
        globals={
            "f" : f,
            "args" : args,
            "kwargs" : kwargs,
        }
    )

    return t0.blocked_autorange(min_run_time=MIN_RUNTIME)