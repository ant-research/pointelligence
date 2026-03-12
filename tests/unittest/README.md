# Dependency on [KeOps](https://www.kernel-operations.io/keops/index.html)

There is no dependency on KeOps, however, we include it in our unittest for comparisons.

In case there are some compilation errors from pykeops, run the following command first. It may resolve some problem, not sure why.
```console
python -c 'import pykeops; pykeops.test_torch_bindings()'
```