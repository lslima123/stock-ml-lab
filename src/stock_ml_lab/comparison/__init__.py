"""Local-vs-global benchmark and inference comparison utilities."""

__all__ = ["run_local_vs_global_benchmark"]


def __getattr__(name: str):
    if name == "run_local_vs_global_benchmark":
        from stock_ml_lab.comparison.benchmark import run_local_vs_global_benchmark
        return run_local_vs_global_benchmark
    raise AttributeError(name)
