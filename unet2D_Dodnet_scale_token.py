import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicHead(nn.Module):
    """
    The 'Shape-Shifting' Output Layer.
    It takes the image features and a set of dynamically generated parameters (weights/biases)
    and applies them to produce the final segmentation.
    """

    def __init__(self, in_channels=8, hidden_dim=8, num_classes=2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Calculate the number of parameters needed for a 3-layer MLP head
        # Layer 1: Conv2d(in, hidden, 1x1)
        self.w1 = in_channels * hidden_dim
        self.b1 = hidden_dim

        # Layer 2: Conv2d(hidden, hidden, 1x1)
        self.w2 = hidden_dim * hidden_dim
        self.b2 = hidden_dim

        # Layer 3: Conv2d(hidden, out, 1x1)
        self.w3 = hidden_dim * num_classes
        self.b3 = num_classes

        self.total_params = sum([self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])

    def forward(self, x, params):
        """
        x: Feature map [Batch, In_Channels, H, W]
        params: Dynamic weights [Batch, Total_Params]
        """
        B, C, H, W = x.shape

        # We reshape the input to treat the entire batch as one giant "channel" group
        # This allows us to use Grouped Convolution to apply different weights to each image
        x_reshaped = x.reshape(1, B * C, H, W)

        # --- Parse Parameters ---
        # We slice the flat 'params' vector into weights and biases for each layer
        curr = 0

        def get_w_b(num_w, shape_w, num_b):
            nonlocal curr
            # Slice Weight
            w = params[:, curr : curr + num_w].reshape(B * shape_w[0], shape_w[1], 1, 1)
            curr += num_w
            # Slice Bias
            b = params[:, curr : curr + num_b].reshape(B * shape_w[0])
            curr += num_b
            return w, b

        # Layer 1
        w1, b1 = get_w_b(self.w1, (self.hidden_dim, self.in_channels), self.b1)
        # Layer 2
        w2, b2 = get_w_b(self.w2, (self.hidden_dim, self.hidden_dim), self.b2)
        # Layer 3
        w3, b3 = get_w_b(self.w3, (self.num_classes, self.hidden_dim), self.b3)

        # --- Dynamic Forward Pass ---
        # Groups=B ensures that Image[0] is convolved with Weights[0], Image[1] with Weights[1], etc.

        out = F.conv2d(x_reshaped, w1, bias=b1, groups=B)
        out = F.relu(out)

        out = F.conv2d(out, w2, bias=b2, groups=B)
        out = F.relu(out)

        out = F.conv2d(out, w3, bias=b3, groups=B)

        # Reshape back to [Batch, Num_Classes, H, W]
        return out.reshape(B, self.num_classes, H, W)


class ResBlock(nn.Module):
    """
    Standard Residual Block with GroupNorm (better for small batches than BatchNorm).
    Pre-activation style: Norm -> ReLU -> Conv
    """

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.gn1 = nn.GroupNorm(16, in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )

        self.gn2 = nn.GroupNorm(16, planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.GroupNorm(16, in_planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        residual = x

        out = self.gn1(x)
        out = F.relu(out, inplace=True)
        out = self.conv1(out)

        out = self.gn2(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        return out + residual


class unet2D(nn.Module):
    def __init__(self, num_classes=4, num_scale=1, weight_std=False):
        super().__init__()
        self.inplanes = 32

        # --- Backbone Configuration ---
        # Channels at each stage: [32, 64, 128, 256, 256]
        self.stage_dims = [32, 64, 128, 256, 256]

        # 1. Initial Conv
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)

        # 2. Encoder Layers (ResBlocks)
        # We manually define them to match the original architecture
        self.layer0 = self._make_layer(32, 32, stride=1)
        self.layer1 = self._make_layer(32, 64, stride=2)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 256, stride=2)

        # 3. Fusion & Controller
        self.fusionConv = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
        )

        self.GAP = nn.Sequential(
            nn.GroupNorm(16, 256), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1))
        )

        # The Dynamic Head
        self.dynamic_head = DynamicHead(in_channels=8, hidden_dim=8, num_classes=2)

        # The Controller: Predicts the weights for the Dynamic Head
        # Input: 256 (GAP features) + Token Embedding Size
        # Output: Total parameters needed by dynamic_head
        self.controller = nn.Conv2d(256, self.dynamic_head.total_params, kernel_size=1)

        # 4. Decoder Layers
        self.upsamplex2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

        self.x8_resb = self._make_layer(256, 128, stride=1)
        self.x4_resb = self._make_layer(128, 64, stride=1)
        self.x2_resb = self._make_layer(64, 32, stride=1)
        self.x1_resb = self._make_layer(32, 32, stride=1)

        self.precls_conv = nn.Sequential(
            nn.GroupNorm(16, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 8, kernel_size=1),  # Reduces to 8 channels for the head
        )

        # 5. TOKENS (The "Switch")
        # We need a token for every stage of the encoder + the head
        # Total Token Dim = 32+64+128+256+256 + 256(Head) = 992
        self.token_dims = [32, 64, 128, 256, 256, 256]
        total_token_dim = sum(self.token_dims)

        self.cls_emb = nn.Parameter(torch.randn(1, num_classes, total_token_dim))
        self.sls_emb = nn.Parameter(torch.randn(1, num_scale, total_token_dim))

    def _make_layer(self, inplanes, planes, stride=1):
        # Simplified block creation
        return ResBlock(inplanes, planes, stride)

    def forward(self, x, task_id, scale_id):
        B = x.shape[0]

        # --- 1. Retrieve Tokens ---
        # Fetch Class Token
        # Shape: [B, Total_Dim]
        c_token = self.cls_emb[0, task_id.long(), :]

        # Fetch Scale Token
        s_token = self.sls_emb[0, scale_id.long(), :]

        # Combine
        tokens = c_token + s_token  # [B, 992]

        # Slice tokens for each layer
        # We reshape them to [B, Dim, 1, 1] so we can add them to feature maps
        layer_tokens = []
        curr = 0
        for dim in self.token_dims:
            t = tokens[:, curr : curr + dim].unsqueeze(-1).unsqueeze(-1)
            layer_tokens.append(t)
            curr += dim

        # --- 2. Encoder (With Token Injection) ---
        # Note: Original code added tokens BEFORE the layer.

        x = self.conv1(x)

        # Layer 0
        x = self.layer0(x + layer_tokens[0])  # Inject Token 0
        skip0 = x

        # Layer 1
        x = self.layer1(x + layer_tokens[1])  # Inject Token 1
        skip1 = x

        # Layer 2
        x = self.layer2(x + layer_tokens[2])  # Inject Token 2
        skip2 = x

        # Layer 3
        x = self.layer3(x + layer_tokens[3])  # Inject Token 3
        skip3 = x

        # Layer 4
        x = self.layer4(x + layer_tokens[4])  # Inject Token 4

        # --- 3. Controller & Fusion ---
        x = self.fusionConv(x)

        # Global Pooling
        feat_gap = self.GAP(x)  # [B, 256, 1, 1]

        # Controller Input: Image Features + Head Token (Last Token)
        # Note: We add the HEAD token here to condition the weight generation
        controller_input = feat_gap + layer_tokens[5]

        # Generate Parameters
        params = self.controller(controller_input)  # [B, 162, 1, 1]
        params = params.flatten(1)  # [B, 162]

        # --- 4. Decoder ---
        # Up x8 (from Layer 4 -> Layer 3 size)
        x = self.upsamplex2(x)
        x = x + skip3
        x = self.x8_resb(x)

        # Up x4
        x = self.upsamplex2(x)
        x = x + skip2
        x = self.x4_resb(x)

        # Up x2
        x = self.upsamplex2(x)
        x = x + skip1
        x = self.x2_resb(x)

        # Up x1
        x = self.upsamplex2(x)
        x = x + skip0
        x = self.x1_resb(x)

        # Prepare for Head
        head_input = self.precls_conv(x)  # [B, 8, H, W]

        # --- 5. Dynamic Head ---
        # Apply the generated parameters to the image features
        logits = self.dynamic_head(head_input, params)

        return logits


def UNet2D(num_classes=4, num_scale=1, weight_std=False):
    return unet2D(num_classes, num_scale, weight_std)
