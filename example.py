import numpy as np
import torch

from time import process_time
from prdc import compute_prdc, compute_prdc_torch


num_real_samples = num_fake_samples = 10000
feature_dim = 1000
nearest_k = 5
real_features = np.random.normal(loc=0.0, scale=1.0,
                                 size=[num_real_samples, feature_dim])

fake_features = np.random.normal(loc=0.0, scale=1.0,
                                 size=[num_fake_samples, feature_dim])
real_features_torch = torch.Tensor(real_features)
fake_features_torch = torch.Tensor(fake_features)

start_time = process_time()
metrics = compute_prdc(real_features=real_features,
                       fake_features=fake_features,
                       nearest_k=nearest_k)
print('total time (numpy/CPU): ' + str(process_time() - start_time))
print(metrics)

start_time = process_time()
metrics = compute_prdc_torch(real_features=real_features_torch,
                       fake_features=fake_features_torch,
                       nearest_k=nearest_k)
print('total time (torch/CPU): ' + str(process_time() - start_time))
print(metrics)

if torch.cuda.is_available():
    print('profiling in GPU')
    real_features_torch_gpu = real_features_torch.cuda()
    fake_features_torch_gpu = fake_features_torch.cuda()
    start_time = process_time()
    metrics = compute_prdc_torch(real_features=real_features_torch_gpu,
                           fake_features=fake_features_torch_gpu,
                           nearest_k=nearest_k)
    torch.cuda.synchronize()
    print('total time (torch/GPU): ' + str(process_time() - start_time))
    print(metrics)
else:
    print('cuda GPU not available')
