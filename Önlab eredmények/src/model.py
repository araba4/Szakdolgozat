import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPricer(nn.Module):

    def __init__(self, input_dim: int = 4, hidden_dim: int = 100, n_layers: int = 4):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepMLPPricer(nn.Module):

    def __init__(self, input_dim: int = 4, hidden_dim: int = 256,
                 n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        blocks = []
        for _ in range(n_layers):
            blocks.append(nn.LayerNorm(hidden_dim))
            blocks.append(nn.ReLU())
            blocks.append(nn.Dropout(dropout))
            blocks.append(nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.Sequential(*blocks)

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.output_norm(x)
        return self.output_proj(x)


class _ResidualBlock(nn.Module):

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNetPricer(nn.Module):

    def __init__(self, input_dim: int = 4, hidden_dim: int = 256,
                 n_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[_ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.output_proj(x)


class _GELUResidualBlock(nn.Module):

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class GELUResNetPricer(nn.Module):

    def __init__(self, input_dim: int = 4, hidden_dim: int = 256,
                 n_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[_GELUResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.output_norm(x)
        return self.output_proj(x)


class DenseMLPPricer(nn.Module):

    def __init__(self, input_dim: int = 4, hidden_dim: int = 128,
                 n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        in_dim = input_dim
        for _ in range(n_layers):
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim += hidden_dim

        final_dim = input_dim + n_layers * hidden_dim
        self.output_proj = nn.Linear(final_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [x]
        for layer in self.layers:
            h_in = torch.cat(outputs, dim=-1)
            h = self.dropout(F.gelu(layer(h_in)))
            outputs.append(h)
        return self.output_proj(torch.cat(outputs, dim=-1))


class _HighwayBlock(nn.Module):

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.H = nn.Linear(dim, dim)
        self.T = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.constant_(self.T.bias, -1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H = self.dropout(F.gelu(self.H(x)))
        T = torch.sigmoid(self.T(x))
        return H * T + x * (1.0 - T)


class HighwayPricer(nn.Module):

    def __init__(self, input_dim: int = 4, hidden_dim: int = 256,
                 n_blocks: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [_HighwayBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.input_proj(x))
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


class FINNPricer(nn.Module):

    def __init__(self, input_dim: int = 4, approx_dim: int = 64,
                 resnet_dim: int = 256, n_blocks: int = 3, dropout: float = 0.1):
        super().__init__()

        self.approx = nn.Sequential(
            nn.Linear(input_dim, approx_dim),
            nn.GELU(),
            nn.Linear(approx_dim, approx_dim),
            nn.GELU(),
            nn.Linear(approx_dim, 1),
        )

        self.correction_proj = nn.Sequential(
            nn.Linear(input_dim, resnet_dim),
            nn.GELU(),
        )
        self.correction_blocks = nn.Sequential(
            *[_GELUResidualBlock(resnet_dim, dropout) for _ in range(n_blocks)]
        )
        self.correction_norm = nn.LayerNorm(resnet_dim)
        self.correction_out = nn.Linear(resnet_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs_approx = self.approx(x)
        delta = self.correction_proj(x)
        delta = self.correction_blocks(delta)
        delta = self.correction_norm(delta)
        delta = self.correction_out(delta)
        return bs_approx + delta


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(name: str, **kwargs) -> nn.Module:
    registry = {
        'mlp':          MLPPricer,
        'deep_mlp':     DeepMLPPricer,
        'resnet':       ResNetPricer,
        'gelu_resnet':  GELUResNetPricer,
        'dense_mlp':    DenseMLPPricer,
        'highway':      HighwayPricer,
        'finn':         FINNPricer,
    }
    if name not in registry:
        raise ValueError(f"Ismeretlen modell: '{name}'. Válasszon: {list(registry)}")
    return registry[name](**kwargs)
