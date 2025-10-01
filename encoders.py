import math
import torch
import torch.nn as nn
import torch.nn.init as init


def initialize_weights(m):
    """
    Applies different weight initialization methods based on layer type.
    """
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)  # Xavier for fully connected layers
        if m.bias is not None:
            init.zeros_(m.bias)  # Zero bias initialization

    elif isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Kaiming for conv layers
        if m.bias is not None:
            init.constant_(m.bias, 0)  # Zero bias

    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)  # Initialize BN scale factor to 1
        init.constant_(m.bias, 0)  # Zero bias

    elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                init.xavier_uniform_(param.data)  # Xavier for input weights
            elif "weight_hh" in name:
                init.orthogonal_(param.data)  # Orthogonal for hidden state
            elif "bias" in name:
                param.data.fill_(0)  # Zero biases


class AppearanceTransform(nn.Module):
    def __init__(self, global_appear_dim, local_appear_dim, in_dim=3):
        super().__init__()
        self.in_dim = in_dim

        self.mlp = nn.Sequential(
            nn.Linear(global_appear_dim + local_appear_dim + in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, in_dim*2),
        )

    def forward(self, color, global_encoding, local_encoding, viewdir=None):
        del viewdir  # Viewdirs interface is kept to be compatible with prev. version

        encoding_input = torch.cat((color, global_encoding, local_encoding), dim=-1).cuda()
        offset, mul = torch.split(self.mlp(encoding_input), [color.shape[-1], color.shape[-1]], dim=-1)
        mul = mul * 0.1 + 1
        return color * mul + offset


# Local encoding
def _get_fourier_features(xyz, num_features=3):
    # xyz = torch.from_numpy(xyz).to(dtype=torch.float32)
    xyz = xyz - xyz.mean(dim=0, keepdim=True)
    xyz = xyz / torch.quantile(xyz.abs(), 0.97, dim=0) * 0.5 + 0.5
    freqs = torch.repeat_interleave(
        2**torch.linspace(0, num_features-1, num_features, dtype=xyz.dtype, device=xyz.device), 2)
    offsets = torch.tensor([0, 0.5 * math.pi] * num_features, dtype=xyz.dtype, device=xyz.device)
    feat = xyz[..., None] * freqs[None, None] * 2 * math.pi + offsets[None, None]
    feat = torch.flatten(torch.sin(feat), start_dim=1)
    return feat


if __name__ == "__main__":
    # Example usage
    appearance_transform = AppearanceTransform(global_appear_dim=64, local_appear_dim=64)
    color = torch.randn(4, 3)
    global_encoding = torch.randn(4, 64)
    local_encoding = torch.randn(4, 64)
    transformed_color = appearance_transform(color, global_encoding, local_encoding)
    print(transformed_color.shape)

    xyz = torch.randn(100, 3)
    position_encoding = _get_fourier_features(xyz, num_features=3)
    print(position_encoding.shape)

