import torch
import numpy as np
from sklearn.neighbors import KDTree
from hausdorffCuda import HD
import time

def t2n(t):
    return t.detach().cpu().numpy()

def run_kdtree(ref, query, scale=1, offset=0):
    ref = ref / scale - offset
    B = ref.shape[0]

    hausdorff = [0.0 for _ in range(B)]
    # one direction
    for b in range(B):
        query_kd_tree = KDTree(query[b])
        one_distances, _ = query_kd_tree.query(ref[b])
        ref_to_query_hausdorff = np.max(one_distances)

        # other direction
        ref_kd_tree = KDTree(ref[b])
        two_distances, _ = ref_kd_tree.query(query[b])
        query_to_ref_hausdorff = np.max(two_distances)

        hausdorff[b] = max(ref_to_query_hausdorff, query_to_ref_hausdorff)
    
    return np.array(hausdorff)

def run_HausdorffCuda(ref, query):
    ref = torch.from_numpy(ref).float().cuda()
    query = torch.from_numpy(query).float().cuda()
    hausdorffDist = HD(transpose_mode=True)
    d  = hausdorffDist(ref, query)
    return t2n(d)

def compare(dim, n1, n2=-1):
    if n2 < 0:
        n2 = n1

    kd_test_times = []
    cuda_test_times = []
    for _ in range(5):
        ref = np.random.random((2, n1, dim))
        query = np.random.random((2, n2, dim))

        t0 = time.perf_counter()
        hausdorff_dist = run_kdtree(ref, query)
        t1 = time.perf_counter()
        kd_test_times.append(t1-t0)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cuda_dist = run_HausdorffCuda(ref, query)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        cuda_test_times.append(t1-t0)

        np.testing.assert_allclose(hausdorff_dist, cuda_dist, rtol=5e-2, atol=5e-3)
    
    kd_test_times = np.array(kd_test_times)
    cuda_test_times = np.array(cuda_test_times)
    print("KDTree times:", np.mean(kd_test_times))
    print("CUDA times:", np.mean(cuda_test_times))

class TestHausdorffCuda:

    def test_hausdorff_cuda_performance(self, benchmark):
        dim = 5
        ref = np.random.random((1, 224, dim))
        query = np.random.random((1, 224, dim))
        benchmark(run_HausdorffCuda, ref, query)

    def test_hausdorff_cuda_5_1000(self):
        compare(5, 1000)

    def test_hausdorff_cuda_5_100(self):
        compare(5, 100)

    def test_hausdorff_cuda_5_10(self):
        compare(5, 10)

    def test_hausdorff_cuda_5_1001(self):
        compare(5, 1001)

    def test_hausdorff_cuda_5_101(self):
        compare(5, 101)

    def test_hausdorff_cuda_5_11(self):
        compare(5, 11)

    def test_hausdorff_cuda_5_300000_50(self):
        compare(5, 30000, 50)

    def test_hausdorff_cuda_5_300001_50(self):
        compare(5, 30001, 50)

    def test_hausdorff_cuda_5_10000(self):
        compare(5, 10000)

    def test_hausdorff_cuda_5_10001(self):
        compare(5, 10001)