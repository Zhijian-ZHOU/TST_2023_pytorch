import numpy as np
import torch
from utils_Local import Local_OPT, density_estimate_test, sample_BLOB, explore_regions, TEST_density_diff
import warnings
warnings.filterwarnings("ignore")

check = 1
dtype = torch.float
KK = 10  # number of trails
K = 10 # number of test
alpha = 0.05  # significant level
beta = 2.0

np.random.seed(1102)
torch.manual_seed(1102)
torch.cuda.manual_seed(1102)
torch.backends.cudnn.deterministic = True
is_cuda = False
device = torch.device("cpu")

Results = np.zeros([2])
Middle_results = np.zeros([KK, K])

N = 10000
min_percent = 0.015625  ###   1 / s with s = 64
check_percent = 0.0625  ### min_percent * 4
split_adjust = 1.2   ## avoid empty rectangle region

N_epoch = 1000
learning_rate = 0.007

n_Anchors = 17
batch_size = 10000
NUM = int(check_percent / min_percent) ## the number of most different rectangle regions

X, Y = sample_BLOB(N = 500000, rs = 123, check = check)
S = np.concatenate((X, Y), axis=0)
density1, density2 = density_estimate_test(S, 500000)

for kk in range(KK):
    X_tr, Y_tr = sample_BLOB(N = int(N/2), rs = 123 * kk + 3, check = check)

    """Train"""
    Anchors, gwidths, M_matrixs, Tree, T_level = Local_OPT(np.concatenate((X_tr, Y_tr), axis=0), len(X_tr), n_Anchors, N_epoch, learning_rate, min_percent, split_adjust, 123 * kk + 3, device, dtype, batch_size = batch_size)

    for k in range(K):
        X_te, Y_te = sample_BLOB(N=int(N / 2), rs=321 * kk + 110, check=check)
        """identify the index set of local significant differences"""
        DIFF_idx = explore_regions(np.concatenate((X_te, Y_te), axis=0), len(X_te), Anchors, gwidths, M_matrixs, T_level, Tree.copy(), alpha, beta, split_adjust, device, dtype, 20)

        mean_diffs_ours = TEST_density_diff(S, 500000, Anchors, gwidths, M_matrixs, T_level, Tree.copy(), DIFF_idx[:NUM], split_adjust, density1, density2, device, dtype)
        # Gather results
        Middle_results[kk][k] = mean_diffs_ours

Results[0] = np.sum([middle.sum() / float(K) for middle in Middle_results]) / float(KK)
Results[1] = np.std([middle.sum() / float(K) for middle in Middle_results])

print("Mean and std of DIFFS: ", str(Results[0]), str(Results[1])) ## output the original value of density difference without normalization
