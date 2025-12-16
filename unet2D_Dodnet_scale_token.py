import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicHead(nn.Module):
    """
    The 'Shape-Shifting' Output Layer.
    """

    def __init__(self, in_channels=8, hidden_dim=8, num_classes=2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

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
        B, C, H, W = x.shape
        x_reshaped = x.reshape(1, B * C, H, W)

        curr = 0

        def get_w_b(num_w, shape_w, num_b):
            nonlocal curr
            w = params[:, curr : curr + num_w].reshape(B * shape_w[0], shape_w[1], 1, 1)
            curr += num_w
            b = params[:, curr : curr + num_b].reshape(B * shape_w[0])
            curr += num_b
            return w, b

        w1, b1 = get_w_b(self.w1, (self.hidden_dim, self.in_channels), self.b1)
        w2, b2 = get_w_b(self.w2, (self.hidden_dim, self.hidden_dim), self.b2)
        w3, b3 = get_w_b(self.w3, (self.num_classes, self.hidden_dim), self.b3)

        out = F.conv2d(x_reshaped, w1, bias=b1, groups=B)
        out = F.relu(out)
        out = F.conv2d(out, w2, bias=b2, groups=B)
        out = F.relu(out)
        out = F.conv2d(out, w3, bias=b3, groups=B)

        return out.reshape(B, self.num_classes, H, W)


class ResBlock(nn.Module):
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

        # 1. Encoder Layers
        # Initial Conv: 3 -> 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)

        # Layer 0: 32 -> 32 (Stride 1)
        self.layer0 = self._make_layer(32, 32, stride=1)

        # Layer 1: 32 -> 64 (Stride 2) -- Note: Input is 32!
        self.layer1 = self._make_layer(32, 64, stride=2)

        # Layer 2: 64 -> 128 (Stride 2)
        self.layer2 = self._make_layer(64, 128, stride=2)

        # Layer 3: 128 -> 256 (Stride 2)
        self.layer3 = self._make_layer(128, 256, stride=2)

        # Layer 4: 256 -> 256 (Stride 2)
        self.layer4 = self._make_layer(256, 256, stride=2)

        # 2. Dynamic Head Components
        self.fusionConv = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
        )
        self.GAP = nn.Sequential(
            nn.GroupNorm(16, 256), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.dynamic_head = DynamicHead(in_channels=8, hidden_dim=8, num_classes=2)
        self.controller = nn.Conv2d(256, self.dynamic_head.total_params, kernel_size=1)

        # 3. Decoder
        self.upsamplex2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.x8_resb = self._make_layer(256, 128, stride=1)
        self.x4_resb = self._make_layer(128, 64, stride=1)
        self.x2_resb = self._make_layer(64, 32, stride=1)
        self.x1_resb = self._make_layer(32, 32, stride=1)

        self.precls_conv = nn.Sequential(
            nn.GroupNorm(16, 32), nn.ReLU(inplace=True), nn.Conv2d(32, 8, kernel_size=1)
        )

        # 4. TOKENS
        # FIX: The dimensions must match the INPUT of the layer they are added to.
        # Layer0 Input: 32 (From conv1) -> Token Dim: 32
        # Layer1 Input: 32 (From layer0) -> Token Dim: 32
        # Layer2 Input: 64 (From layer1) -> Token Dim: 64
        # Layer3 Input: 128 (From layer2) -> Token Dim: 128
        # Layer4 Input: 256 (From layer3) -> Token Dim: 256
        # Head Input: 256 (GAP) -> Token Dim: 256
        self.token_dims = [32, 32, 64, 128, 256, 256]
        total_token_dim = sum(self.token_dims)

        self.cls_emb = nn.Parameter(torch.randn(1, num_classes, total_token_dim))
        self.sls_emb = nn.Parameter(torch.randn(1, num_scale, total_token_dim))

    def _make_layer(self, inplanes, planes, stride=1):
        return ResBlock(inplanes, planes, stride)

    def forward(self, x, task_id, scale_id):
        B = x.shape[0]

        # 1. Tokens
        c_token = self.cls_emb[0, task_id.long(), :]
        s_token = self.sls_emb[0, scale_id.long(), :]
        tokens = c_token + s_token

        layer_tokens = []
        curr = 0
        for dim in self.token_dims:
            t = tokens[:, curr : curr + dim].unsqueeze(-1).unsqueeze(-1)
            layer_tokens.append(t)
            curr += dim

        # 2. Encoder
        x = self.conv1(x)

        # Layer 0 (Input 32 + Token 32)
        x = self.layer0(x + layer_tokens[0])
        skip0 = x

        # Layer 1 (Input 32 + Token 32)
        x = self.layer1(x + layer_tokens[1])
        skip1 = x

        # Layer 2 (Input 64 + Token 64)
        x = self.layer2(x + layer_tokens[2])
        skip2 = x

        # Layer 3 (Input 128 + Token 128)
        x = self.layer3(x + layer_tokens[3])
        skip3 = x

        # Layer 4 (Input 256 + Token 256)
        x = self.layer4(x + layer_tokens[4])

        # 3. Controller
        x = self.fusionConv(x)
        feat_gap = self.GAP(x)
        controller_input = feat_gap + layer_tokens[5]
        params = self.controller(controller_input).flatten(1)

        # 4. Decoder
        x = self.upsamplex2(x) + skip3
        x = self.x8_resb(x)

        x = self.upsamplex2(x) + skip2
        x = self.x4_resb(x)

        x = self.upsamplex2(x) + skip1
        x = self.x2_resb(x)

        x = self.upsamplex2(x) + skip0
        x = self.x1_resb(x)

        head_input = self.precls_conv(x)
        logits = self.dynamic_head(head_input, params)

        return logits


def UNet2D(num_classes=4, num_scale=1, weight_std=False):
    return unet2D(num_classes, num_scale, weight_std)
