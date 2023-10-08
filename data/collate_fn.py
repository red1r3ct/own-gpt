import torch

from lib.encoders import pad_token

def make_collate_fn(context_window, device, min_sample_tokens=10):
    def collate(batch):
        x_padded = []
        y_padded = []
        for data in batch:
            s = data['text']
            start = torch.randint(low=0, high=len(s) - min_sample_tokens, size=(1,)).item()
            end = min(context_window + start, len(s) - 1)
            x = s[start : end - 1]
            y = s[start + 1 : end]
            if len(x) < context_window:
                x = torch.nn.functional.pad(x, (0, context_window - len(x)), mode='constant', value=pad_token)
            if len(y) < context_window:
                y = torch.nn.functional.pad(y, (0, context_window - len(y)), mode='constant', value=pad_token)
                
            x_padded.append(x)
            y_padded.append(y)
            
        return torch.stack(x_padded, dim=0).to(device), torch.stack(y_padded, dim=0).to(device), 

    return collate