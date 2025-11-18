#include <torch/extension.h> //allows C++ talk to pytorch

/*
TODO:
- Implement CPU fallback version of RMSD (for debugging)
- Bind CUDA kernel via at::Tensor inputs
- Ensure float32 contiguous inputs
- Support batched RMSD for faster evaluation
- Expose function: rmsd(A: [B,N,3], B: [B,N,3]) -> [B]
*/

// function declaration in C++ to be called from Python
/*
python equivalent:
def rmsd_cuda(A: Tensor, B: Tensor) -> Tensor:
    return rmsd_cuda_impl(A, B) 
    or return Tensor 

torch::Tensor is pytorch tensor type in c++
put return type as torch::Tensor before the function name

you can call the function from python by importing the module and calling the function
import rmsd_cuda
rmsd_cuda.rmsd(A, B)


Later, you will implement it in a .cu or .cpp file, like:
torch::Tensor rmsd_cuda(torch::Tensor A, torch::Tensor B) {
    // CUDA kernel calls go here
}
*/
torch::Tensor rmsd_cuda(torch::Tensor A, torch::Tensor B);

// 	•	PYBIND11_MODULE exposes C++ functions to Python like a normal module.
// 	•	TORCH_EXTENSION_NAME is a macro that defines the name of the extension module.
// 	•	m is the module object that you will use to define the functions.
// this is the bridge between C++ and Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        /*
        This exposes a C++ function to Python.
        •	"rmsd" → the name of the function in Python
    (this is what Python code will call)
        •	&rmsd_cuda → pointer to your C++ function
    (the one declared above)
        •	"RMSD CUDA kernel (batched, WIP)" → the docstring shown in Python

    So this creates:
    - A Python function rmsd_cuda.rmsd
    that internally calls your C++ function rmsd_cuda.

    in python, you can call the function like this:
    import rmsd_cuda
    rmsd_cuda.rmsd(A, B)

    */
    m.def("rmsd", &rmsd_cuda, "RMSD CUDA kernel (batched, WIP)");
}