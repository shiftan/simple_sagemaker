import torch
num_devices = torch.cuda.device_count()
print(f"Number of Cuda devices: {num_devices}")
for i in range(num_devices):
    print(f"Device: {i} {torch.cuda.get_device_properties(0)}")
