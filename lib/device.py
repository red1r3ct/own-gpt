import torch

from contextlib import nullcontext

class Device:
    def __init__(self):
        self.device_type = detect_device()

    def get_device_type(self):
        return self.device_type

    def get_data_type(self):
        if self.device_type == 'gpu':
            return 'bfloat16'
        return 'float16'
    
    def get_context(self):
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.get_data_type()]
        if self.get_data_type() == 'gpu':
            return torch.amp.autocast(device_type=self.get_data_type(), dtype=ptdtype)
        return nullcontext()

def detect_device():
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'
