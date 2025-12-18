# CHAMFER_CUDA

This repository provides a CUDA-accelerated implementation of the Hausdorff Distance for efficient point cloud distance/similarity computation in Python, integrated with PyTorch.

author: [sunrit2022pers@gmail.com](sunrit2022pers@gmail.com)

### Performance

+ dim   = 5
+ loop  = 5
+ Intel(R) Xeon(R) CPU @ 2.20GHz
+ Tesla T4

| PC1     | PC2     | sklearn  | CUDA     | SpeedUpX |
| :---:   | :---:   | :---:    | :---:    | :---:    |
| 100     | 100     | 1.281 ms | 0.592 ms | 2.16X    |
| 1000    | 1000    | 14.64 ms | 3.535 ms | 4.14X    |
| 10000   | 10000   | 438.6 ms | 27.76 ms | 15.8X    |
| 30000   | 50      | 76.97 ms | 37.54 ms | 2.05X    |

### Installation

```bash
git clone https://github.com/sunrit-noob/HAUSDORFF_CUDA.git
cd HAUSDORFF_CUDA
make && make install
```

### Usage

```python
import torch

# Cuda needs to be available
assert torch.cuda.is_available()

from hausdorffCuda import HD
"""
if transpose_mode is True, 
    pc1 is Tensor [bs x nr x dim]
    pc2 is Tensor [bs x nq x dim]
    
    return 
        hd_dist is Tensor [bs]
else
    pc1 is Tensor [bs x dim x nr]
    pc2 is Tensor [bs x dim x nq]
    
    return 
        hd_dist is Tensor [bs]
"""

hd = HD(transpose_mode=True)

pc1 = torch.rand(32, 1000, 5).cuda()
pc2 = torch.rand(32, 50, 5).cuda()

hd_dist = hd(pc1, pc2)  # 32
```

Try it on Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hcd_2k104Odl9PEOwdR5xFUT72Lf9RlF?usp=sharing)

*Note: Please ensure the Colab runtime is set to GPU (Runtime > Change runtime type > Hardware accelerator)*

### Inspirations

The structure of this repository is inspired from the structure of [KNN_CUDA](https://github.com/unlimblue/KNN_CUDA)