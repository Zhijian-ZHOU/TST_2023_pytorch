import numpy as np
import torch
from utils_Local import Local_OPT, Local_TEST, sample_BLOB
import warnings
warnings.filterwarnings("ignore")

KK = 10 # number of trails
K = 100 # number of test

dtype = torch.float
np.random.seed(1102)
torch.manual_seed(1102)
is_cuda = False
device=torch.device("cpu")
Results = np.zeros([2])
Middle_results = np.zeros([KK, K])

alpha = 0.05 # significant level
n_Anchors = 17 # number of test locations
beta = 2.0
N_epoch = 1000 # optimization epoch
learning_rate = 0.007
batch_size = 900

"""check = 0 for type-I error; check = 1 for test power"""
check = 1
N = 900

for kk in range(KK):
    X_tr, Y_tr = sample_BLOB(N = int(N/2), rs = 123 * (kk + 330), check = check)
    S_tr = np.concatenate((X_tr, Y_tr), axis=0)

    """Train"""
    Anchors, gwidths, M_matrixs, Tree, T_level = Local_OPT(S=S_tr, N1=len(X_tr), n_Anchors=n_Anchors, N_epoch=N_epoch, learning_rate=learning_rate, percent = 1, split_adjust = 1, seed = 123 * (kk + 330), device=device, dtype=dtype, batch_size=batch_size)

    for k in range(K):
        X_te, Y_te = sample_BLOB(N = int(N / 8), rs= 321 * (kk + k + 330), check=check)
        S_te = np.concatenate((X_te, Y_te), axis=0)

        """Test"""
        h = Local_TEST(S=S_te, N1=int(len(S_te) / 2), Anchors=Anchors, gwidths=gwidths, M_matrixs=M_matrixs, infer_dire=Tree[0][-1], alpha=alpha, beta=beta, device=device, dtype=dtype)
        Middle_results[kk][k] = h

if check == 0:
    Results[0] = np.sum([middle.sum() / float(K) for middle in Middle_results]) / float(KK)
    Results[1] = np.std([middle.sum() / float(K) for middle in Middle_results])
    print("Mean and std of Type I error: ", str(Results[0]),
          str(Results[1]))
else:
    Results[0] = np.sum([middle.sum() / float(K) for middle in Middle_results]) / float(KK)
    Results[1] = np.std([middle.sum() / float(K) for middle in Middle_results])
    print("Mean and std of Test Power: ", str(Results[0]),
          str(Results[1]))
