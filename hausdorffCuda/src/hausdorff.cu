#include<cstdio>
#include<cuda.h>

#define BLOCK_DIM 256
#define WARP_SIZE 32

__device__ float atomicMaxFloat(float* addr, float value) {
    int* addr_as_i = (int*)addr;
    int old = *addr_as_i, assumed;

    do {
        assumed = old;
        old = atomicCAS(
            addr_as_i,
            assumed,
            __float_as_int(fmaxf(value, __int_as_float(assumed)))
        );
    } while (assumed != old);

    return __int_as_float(old);
}

__global__ void cuComputeHausdorffGlobal(const float* A, int nA, const float* B, int nB, int dim, float* result) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float thread_min = -__FLT_MAX__;
    if(idx < nA) {
        thread_min = __FLT_MAX__;
        for (int j = 0; j < nB; ++j) {
            float d = 0.0f;
            for (int k = 0; k < dim; ++k) {
                float diff = A[idx + k * nA] - B[j + k * nB];
                d += diff * diff;
            }
            thread_min = fminf(thread_min, d);
        }
    }

    unsigned mask = __activemask();
    for(int offset = WARP_SIZE>>1; offset > 0; offset >>=1) {
        float val = __shfl_down_sync(mask, thread_min, offset);
        thread_min = fmaxf(thread_min, val);    
    }

    __shared__ float shared_max[BLOCK_DIM / WARP_SIZE];
    int lane = threadIdx.x & (WARP_SIZE-1);
    int warp_id = threadIdx.x / WARP_SIZE;

    if(lane == 0) {
        shared_max[warp_id] = thread_min;
    }
    __syncthreads();

    if(warp_id == 0) {
        const int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        float block_max = (lane < num_warps) ? shared_max[lane] : -__FLT_MAX__;
        mask = __activemask();
        for(int offset = WARP_SIZE>>1; offset > 0; offset >>=1) {
            float val = __shfl_down_sync(mask, block_max, offset);
            block_max = fmaxf(block_max, val);    
        }
        if(lane==0) {
            atomicMaxFloat(result, block_max);
        }
    }
}

void hausdorff_distance_device(float* pc1_dev, int pc1_nb, float* pc2_dev, int pc2_nb, 
    int dim, float* result_dev) {
    cudaMemset(result_dev, 0, sizeof(float));
    dim3 block_256(BLOCK_DIM, 1, 1);
    dim3 grid_256_1((pc1_nb + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
    cuComputeHausdorffGlobal<<<grid_256_1, block_256, 0>>>(pc1_dev, pc1_nb, pc2_dev, pc2_nb, dim, result_dev);

    dim3 grid_256_2((pc2_nb + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
    cuComputeHausdorffGlobal<<<grid_256_2, block_256, 0>>>(pc2_dev, pc2_nb, pc1_dev, pc1_nb, dim, result_dev);
}