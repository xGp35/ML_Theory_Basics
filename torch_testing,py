import torch
import time

# Make sure GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Large matrices (increase size if GPU is powerful)
size = 8192

a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

torch.cuda.synchronize()
start = time.time()

for i in range(1000):
    c = torch.matmul(a, b)

    if i % 10 == 0:
        torch.cuda.synchronize()
        print(f"Iteration {i}")

torch.cuda.synchronize()
end = time.time()

print("Finished in:", end - start, "seconds")