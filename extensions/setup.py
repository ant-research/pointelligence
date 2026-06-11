import os
import torch
import glob

from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)


def get_cuda_arch_list():
    if not torch.cuda.is_available():
        print("No CUDA devices found")
        return None

    arch_list = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        # Convert major.minor to string format
        arch = f"{props.major}.{props.minor}"
        if arch not in arch_list:
            arch_list.append(arch)

    return ";".join(arch_list)


# https://en.wikipedia.org/wiki/CUDA
if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    arch_list = get_cuda_arch_list()
    if arch_list is not None:
        arch_list += "+PTX"
        print(f"TORCH_CUDA_ARCH_LIST={arch_list}")
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
    else:
        print("Using default CUDA architecture list for build")

# usage: TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6 8.7 8.9 9.0+PTX" pip install --no-build-isolation -e .

library_name = "sparse_engines_cuda"

if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False


def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.9
        ],
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
            # CUTLASS 3.x / CuTe requires C++17 and relaxed-constexpr.
            # The host compiler picks up -std=c++17 from torch's own flags,
            # but nvcc needs it explicit for the device-side templates.
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            # Suppress noisy CUTLASS-internal warnings ("declared but never
            # referenced" inside template metaprogramming) so the build log
            # stays readable.
            "-Xcompiler=-Wno-unused-but-set-variable",
            "-Xcompiler=-Wno-unused-variable",
        ],
    }
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, library_name, "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))

    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cu")))

    # Vendored CUTLASS 4.3.4. Required by the Tier-2 vvor
    # CUTLASS kernels (vvor_cutlass_sm80.cu etc.). The repo root is two
    # levels up from this setup.py (extensions/).
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    cutlass_root = os.path.join(repo_root, "third_party", "cutlass")
    cutlass_includes = [
        os.path.join(cutlass_root, "include"),
        os.path.join(cutlass_root, "tools", "util", "include"),
        os.path.join(cutlass_root, "examples", "common"),
    ]

    if use_cuda:
        sources += cuda_sources
    else:
        extra_compile_args["cxx"].append("-DWITHOUT_CUDA")

    ext_modules = [
        extension(
            f"{library_name}._C",
            sources,
            include_dirs=cutlass_includes,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=py_limited_api,
        )
    ]

    return ext_modules


setup(
    name=library_name,
    version="0.0.1",
    packages=["sparse_engines_cuda"],  # find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch"],
    description="Sparse Engines Implemented with CUDA Extensions",
    long_description="",
    long_description_content_type="text/markdown",
    url="",
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)
