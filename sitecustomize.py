"""Project-local Python startup hook for Windows DLL lookup.

This ensures repo-root commands like:
  python -c "from llama_cpp import llama_cpp"
can resolve CUDA/llama native DLL dependencies inside the active venv.
"""

from __future__ import annotations

import os
import sys


def _add_windows_dll_dirs() -> None:
    if os.name != "nt":
        return

    try:
        import ctypes
        import site

        base_candidates = []
        for p in site.getsitepackages():
            if os.path.isdir(p):
                base_candidates.append(p)
        base_candidates.append(os.path.join(sys.prefix, "Lib", "site-packages"))

        llama_lib_dir = None
        for base in base_candidates:
            for rel in (
                os.path.join("llama_cpp", "lib"),
                os.path.join("nvidia", "cublas", "bin"),
                os.path.join("nvidia", "cuda_runtime", "bin"),
                os.path.join("nvidia", "cuda_nvrtc", "bin"),
            ):
                dll_dir = os.path.join(base, rel)
                if os.path.isdir(dll_dir):
                    os.add_dll_directory(dll_dir)
                    if rel == os.path.join("llama_cpp", "lib"):
                        llama_lib_dir = dll_dir

        if llama_lib_dir and os.path.isdir(llama_lib_dir):
            for dll_name in (
                "ggml-base.dll",
                "ggml-cpu.dll",
                "ggml-cuda.dll",
                "ggml.dll",
                "llama.dll",
            ):
                dll_path = os.path.join(llama_lib_dir, dll_name)
                if os.path.isfile(dll_path):
                    try:
                        ctypes.CDLL(dll_path)
                    except Exception:
                        pass
    except Exception:
        # Keep startup resilient; runtime imports will show concrete errors.
        pass


_add_windows_dll_dirs()
