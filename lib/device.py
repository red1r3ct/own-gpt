import torch

class Device:
    def __init__(self):
        self.device_type = detect_device()

    def get_device_type(self):
        return self.device_type

    def get_data_type(self):
        return "bfloat16"
    
    
    
def detect_device():
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'
