import torch
import torch.nn as nn

from enums import ModelType


class ResBlock(nn.Module):

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #return self.relu(self.net(x) + x)
        return self.net(x) + x

class VanillaMLP(nn.Module):

    def __init__(self, d: int):
        super().__init__()
        # Dynamic hidden size based on input dimension
        h = 64 if d < 64 else 256
        
        self.net = nn.Sequential(
            nn.Linear(d, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, d) # Output layer (predicts noise magnitude)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ResMLP(nn.Module):

    def __init__(self, d: int):
        super().__init__()
        # Dynamic hidden size based on input dimension
        h = 64 if d < 64 else 256
        
        # Initial projection layer
        self.input_layer = nn.Sequential(
            nn.Linear(d, h),
            nn.ReLU()
        )
        
        # 5 Residual Blocks
        self.res_blocks = nn.Sequential(
            *[ResBlock(h) for _ in range(5)]
        )
        
        # Output projection back to dimension 'd'
        self.output_layer = nn.Linear(h, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.output_layer(x)

class NoiseEvaluator(nn.Module):

    def __init__(self, input_dim: int, model_type: ModelType = ModelType.RES_MLP):
        super().__init__()
        self.model_type = model_type
        
        if model_type == ModelType.VANILLA_MLP:
            self.network = VanillaMLP(input_dim)
        elif model_type == ModelType.RES_MLP:
            self.network = ResMLP(input_dim)
        else:
            raise ValueError(f"Model type {model_type} is not supported.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)