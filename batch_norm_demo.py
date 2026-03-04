import torch
import torch.nn as nn

# import torch
# if torch.backends.mps.is_available():
#     mps_device = torch.device("mps")
#     x = torch.ones(1, device=mps_device)
#     print (x)
# else:
#     print ("MPS device not found.")


torch.set_printoptions(precision=3)

# Mini-batch: 4 samples, 1 feature
x = torch.tensor([[20.0],
                  [25.0],
                  [30.0],
                  [35.0]])

linear = nn.Linear(1, 1, bias=False)
linear.weight.data.fill_(1.0)  # w = 1

y = linear(x)
print(y)

