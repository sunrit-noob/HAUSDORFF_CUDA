#include<vector>
#include<torch/extension.h>
#include<ATen/cuda/CUDAContext.h>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_TYPE(x, t) TORCH_CHECK((x).scalar_type() == (t), #x " must be " #t)
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be on CUDA")
#define CHECK_INPUT(x, t) do { CHECK_CONTIGUOUS(x); CHECK_TYPE(x, t); CHECK_CUDA(x); } while(0)

void hausdorff_distance_device(
    float* pc1_dev, 
    int pc1_nb, 
    float* pc2_dev, 
    int pc2_nb, 
    int dim,  
    float* result_dev
    );

at::Tensor hausdorff_distance(
    at::Tensor & pc1, 
    at::Tensor & pc2
    ){

    CHECK_INPUT(pc1, at::kFloat);
    CHECK_INPUT(pc2, at::kFloat);
    int dim = pc1.size(0);
    int pc1_nb = pc1.size(1);
    int pc2_nb = pc2.size(1);
    float * pc1_dev = pc1.data_ptr<float>();
    float * pc2_dev = pc2.data_ptr<float>();
    auto result = torch::zeros({1}, pc1.options());
    float* result_dev = result.data_ptr<float>();

    hausdorff_distance_device(
        pc1_dev,
        pc1_nb,
        pc2_dev,
        pc2_nb,
        dim,
        result_dev
    );

    return at::sqrt(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hausdorff_distance", &hausdorff_distance, "Hausdorff Distance cuda version");
}