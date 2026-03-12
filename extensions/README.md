# C++/CUDA Extensions.

List of Supported OPs:
* MVMR 
* VVOR
* SDPA (forward and backward)

The extension mechanism is implemented following <https://github.com/pytorch/extension-cpp>. PyTorch 2.4+ required.

To build and install:
```shell
pip install --no-build-isolation -e .
```