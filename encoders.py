import math
import torch
import torch.nn as nn
import torchvision.models as models
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


# class AppearanceEncoder(nn.Module):
#     def __init__(self, backbone="resnet18", output_dim=256, pretrained=True, pooling_window=3):
#         super(AppearanceEncoder, self).__init__()
#
#         # Load the backbone (e.g., ResNet, EfficientNet)
#         if backbone == "resnet18":
#             self.model = models.resnet18(pretrained=pretrained)
#             feature_dim = self.model.fc.in_features
#             self.model = nn.Sequential(*list(self.model.children())[:-2])  # Remove fully connected layers
#         elif backbone == "efficientnet_b0":
#             self.model = models.efficientnet_b0(pretrained=pretrained)
#             feature_dim = self.model.classifier[1].in_features
#             self.model = nn.Sequential(*list(self.model.children())[:-2])  # Remove fully connected layers
#         else:
#             raise ValueError("Unsupported backbone: Choose 'resnet18' or 'efficientnet_b0'.")
#
#         # Adaptive Pooling to handle different input sizes
#         self.global_pool = nn.AdaptiveAvgPool2d((pooling_window, pooling_window))
#
#         # Projection head for feature dimensionality reduction
#         self.fc = nn.Linear(feature_dim * pooling_window * pooling_window, output_dim)
#
#     def forward(self, x):
#         x = self.model(x)  # Feature extraction
#         x = self.global_pool(x)  # Global pooling (NxCx1x1)
#         x = torch.flatten(x, 1)  # Flatten to (NxC)
#         x = self.fc(x)  # Project to output dimension
#         return x


# class AppearanceTransform(nn.Module):
#     def __init__(self, global_encoding_dim, local_encoding_dim, in_dim=3):
#         super().__init__()
#         self.in_dim = in_dim
#
#         self.mlp_var = nn.Sequential(
#             nn.Linear(global_encoding_dim + local_encoding_dim + in_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, in_dim),
#         )
#
#         self.mlp_bias = nn.Sequential(
#             nn.Linear(global_encoding_dim + local_encoding_dim + in_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, in_dim),
#         )
#
#     def forward(self, color, global_encoding, local_encoding, viewdir=None):
#         del viewdir  # Viewdirs interface is kept to be compatible with prev. version
#
#         encoding_input = torch.cat((color, global_encoding, local_encoding), dim=-1)
#         var = self.mlp_var(encoding_input) # "* 0.01" in wildgaussian why?
#         bias = self.mlp_bias(encoding_input)
#         return color * var + bias


# class EmbeddingModel(nn.Module):
#     def __init__(self, config: Config):
#         super().__init__()
#         self.config = config
#         # sh_coeffs = 4**2
#         in_dim = 3
#         if config.appearance_model_sh:
#             in_dim = ((config.sh_degree + 1) ** 2) * 3
#         self.mlp = nn.Sequential(
#             nn.Linear(config.appearance_embedding_dim + in_dim + 6 * self.config.appearance_n_fourier_freqs, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, in_dim*2),
#         )
#
#     def forward(self, gembedding, aembedding, color, viewdir=None):
#         del viewdir  # Viewdirs interface is kept to be compatible with prev. version
#
#         input_color = color
#         if not self.config.appearance_model_sh:
#             color = color[..., :3]
#         inp = torch.cat((color, gembedding, aembedding), dim=-1)
#         offset, mul = torch.split(self.mlp(inp) * 0.01, [color.shape[-1], color.shape[-1]], dim=-1)
#         offset = torch.cat((offset / C0, torch.zeros_like(input_color[..., offset.shape[-1]:])), dim=-1)
#         mul = mul.repeat(1, input_color.shape[-1] // mul.shape[-1])
#         return input_color * mul + offset


class AppearanceTransform(nn.Module):
    def __init__(self, global_encoding_dim, local_encoding_dim, in_dim=3):
        super().__init__()
        self.in_dim = in_dim

        self.mlp = nn.Sequential(
            nn.Linear(global_encoding_dim + local_encoding_dim + in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, in_dim*2),
        )

    def forward(self, color, global_encoding, local_encoding, viewdir=None):
        del viewdir  # Viewdirs interface is kept to be compatible with prev. version

        # encoding_input = torch.cat((color, global_encoding, local_encoding), dim=-1)
        # var = self.mlp_var(encoding_input) # "* 0.01" in wildgaussian why?
        # bias = self.mlp_bias(encoding_input)
        # return color * var + bias
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


# if self.embeddings is not None:
#     embeddings = _get_fourier_features(xyz, num_features=self.config.appearance_n_fourier_freqs)
#     embeddings.add_(torch.randn_like(embeddings) * 0.0001)
#     if not self.config.appearance_init_fourier:
#         embeddings.normal_(0, 0.01)
#     self.embeddings.data.copy_(embeddings)


if __name__ == "__main__":
    # Example usage
    model = AppearanceEncoder(backbone="resnet18", output_dim=256, pretrained=False)
    input_tensor = torch.randn(4, 3, 1224, 624)  # Batch of 4 images with varying sizes
    features = model(input_tensor)  # (4, 256)
    print(features.shape)

    # Example usage
    appearance_transform = AppearanceTransform(global_encoding_dim=64, local_encoding_dim=64)
    color = torch.randn(4, 3)
    global_encoding = torch.randn(4, 64)
    local_encoding = torch.randn(4, 64)
    transformed_color = appearance_transform(color, global_encoding, local_encoding)
    print(transformed_color.shape)

    xyz = torch.randn(100, 3)
    position_encoding = _get_fourier_features(xyz, num_features=3)
    print(position_encoding.shape)

