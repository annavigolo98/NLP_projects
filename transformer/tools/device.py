from pydantic import BaseModel
import torch

class Device(BaseModel):
    @staticmethod
    def get_device():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device