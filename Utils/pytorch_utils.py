import torch

def clear_gpu_cache():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def check_gpu_memory():
    print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"Reserved Memory: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")



