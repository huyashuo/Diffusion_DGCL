import torch
import torch.nn as nn
import torch.nn.functional as F

class RepConvBranch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

class RepRFNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 多尺度分支
        self.branch_3x3 = RepConvBranch(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch_1x1 = RepConvBranch(in_channels, out_channels, kernel_size=1, padding=0)
        self.branch_5x5 = RepConvBranch(in_channels, out_channels, kernel_size=5, padding=2)

        self.fusion = nn.Conv1d(out_channels * 3, out_channels, kernel_size=1)
        self.act = nn.GELU()
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 输入形状 B, T, C → 转为 B, C, T
        x = x.transpose(1, 2)

        # 多尺度特征提取
        x1 = self.branch_3x3(x)
        x2 = self.branch_1x1(x)
        x3 = self.branch_5x5(x)

        # 拼接并融合
        concat = torch.cat([x1, x2, x3], dim=1)  # [B, C*3, T]
        fused = self.fusion(concat)  # [B, C, T]

        # 残差连接 + 激活
        res = self.residual(x)
        out = self.act(fused + res)

        # 恢复为 B, T, C
        return out.transpose(1, 2)
