import time
import torch
import os
import psutil

class PerformanceTracker:
    def __init__(self):
        self.start_time = 0
        self.process = psutil.Process(os.getpid())

    def tracking_start(self):
        self.start_time = time.perf_counter()

        if torch.cuda.is_available():
           torch.cuda.reset_peak_memory_stats()

    def tracking_stop(self, model, model_path):
        dur  = time.perf_counter() - self.start_time

        ram_usage = self.process.memory_info().rss / (1024 ** 2)

        cpu_load = psutil.cpu_percent(interval=None)

        gpu_mem_used = 0
        if torch.cuda.is_available():
            gpu_mem_used = torch.cuda.max_memory_allocated() / (1024 ** 2)

        torch.save(model.state_dict(), model_path)
        model_size = os.path.getsize(model_path) / (1024 ** 2)

        print("\nPerformance report:")
        print(f"Training time: {dur:.2f} s")
        print(f"RAM used: {ram_usage:.2f} MB")
        print(f"CPU load: {cpu_load}%")

        if torch.cuda.is_available():
            print(f"VRAM used: {gpu_mem_used:.2f} MB")

        print(f"1D Model size: {model_size:.4f} MB")
        