from model.nms.nms_wrapper import nms
import torch

a = [
    [10, 10, 20, 20, 0.5],
    [12, 12, 21, 21, 0.4]
]

a = torch.tensor(a, dtype=torch.float32)
keep = nms(a, 0.5, True)
print(keep)