#include <Python.h>
#include <torch/library.h>

extern "C" {
/* Creates a dummy empty _C module that can be imported from Python.
   The import from Python will load the .so consisting of this file
   in this extension, so that the TORCH_LIBRARY static initializers
   below are run. */
PyObject* PyInit__C(void) {
	static struct PyModuleDef module_def = {
		PyModuleDef_HEAD_INIT,
		"_C", /* name of module */
		NULL, /* module documentation, may be NULL */
		-1,   /* size of per-interpreter state of the module,
	             or -1 if the module keeps state in global variables. */
		NULL, /* methods */
	};
	return PyModule_Create(&module_def);
}
}

// Defines the operators
TORCH_LIBRARY(sparse_engines_cuda, m) {
	m.def("sparse_matrix_vector_multiplication_reduction(Tensor a, Tensor a_idx, Tensor b, Tensor b_idx, Tensor o_idx, int n) -> Tensor");
	m.def("sparse_vector_vector_outer_product_reduction(Tensor a, Tensor a_idx, Tensor b, Tensor b_idx, Tensor o_idx, int n) -> Tensor");

	m.def("bucket_arrange(Tensor bucket_indices, int num_buckets) -> (Tensor, Tensor)");
}
