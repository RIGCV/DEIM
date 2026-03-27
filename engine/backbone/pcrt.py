import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PRCT(nn.Module):


    def __init__(self, num_scales=3):
        super(PRCT, self).__init__()
        self.num_scales = num_scales
        self.eps = 1e-6  # 数值稳定性参数

        # 可学习的色彩敏感度参数（针对植物特征优化）
        self.green_sensitivity = nn.Parameter(torch.tensor([0.5]))  # 对绿色通道的敏感度
        self.necrotic_sensitivity = nn.Parameter(torch.tensor([0.5]))  # 对坏死区域的敏感度

        # 多尺度高斯核（用于捕捉不同大小的病斑）
        self.gaussian_kernels = self._create_gaussian_kernels()

        # 生物视觉启发的映射参数
        self.gamma = nn.Parameter(torch.tensor([0.3]))  # 亮度适应参数
        self.beta = nn.Parameter(torch.tensor([1.0]))  # 对比度增强参数

        # 病害特征强化参数
        self.lesion_strength = nn.Parameter(torch.tensor([0.3]))

    def _create_gaussian_kernels(self):
        """创建多尺度高斯核用于病变区域分析"""
        kernels = []
        for s in range(1, self.num_scales + 1):
            sigma = 1.0 + s * 0.8  # 尺度递增的标准差
            kernel_size = int(2 * math.ceil(2 * sigma) + 1)

            # 创建1D高斯核
            kernel_1d = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
            kernel_1d = torch.exp(-0.5 * (kernel_1d / sigma) ** 2)
            kernel_1d = kernel_1d / (kernel_1d.sum() + self.eps)  # 数值保护

            # 扩展为2D高斯核
            kernel_2d = torch.outer(kernel_1d, kernel_1d)
            kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)  # [out, in, H, W]

            kernels.append(kernel_2d)
        return nn.ParameterList([nn.Parameter(k) for k in kernels])

    def _multi_scale_analysis(self, channel):
        """多尺度分析，突出病变区域的纹理特征"""
        channel = torch.clamp(channel, 0.0, 1.0)  # 输入范围保护
        scales = []

        for kernel in self.gaussian_kernels:
            kernel = kernel.to(channel.device).to(channel.dtype)
            filtered = F.conv2d(channel, kernel, padding=kernel.shape[-1] // 2)
            scales.append(filtered)

        # 计算尺度差异（突出病变边缘）
        scale_stack = torch.stack(scales, dim=1)
        scale_var = torch.var(scale_stack, dim=1, keepdim=True)
        scale_diff = torch.clamp(scale_var, 0.0, 1.0)  # 限制方差范围

        # 确保输出维度正确
        scale_diff = torch.squeeze(scale_diff, dim=1)
        if scale_diff.dim() == 3:
            scale_diff = scale_diff.unsqueeze(1)

        return scale_diff

    def _bio_inspired_mapping(self, value):
        """生物视觉映射，增强叶片与病害区域的区分度"""
        value_clamped = torch.clamp(value, self.eps, 1.0 - self.eps)

        # 模拟人眼对绿色植物的亮度适应特性
        adapted = torch.tanh(value_clamped * self.beta)

        # 增强中间亮度范围（植物叶片常见亮度）
        mask = (value_clamped > 0.3) & (value_clamped < 0.7)
        sine_term = torch.sin(value_clamped * math.pi)
        adapted = torch.where(
            mask,
            adapted * (1 + self.gamma * sine_term),
            adapted
        )

        return torch.clamp(adapted, -1.0 + self.eps, 1.0 - self.eps)

    def forward_rgb_to_prct(self, rgb):
        """正向变换：RGB -> PRCT颜色空间"""
        # 1. 输入验证与预处理
        rgb_clamped = torch.clamp(rgb, 0.0, 1.0)  # 确保输入在有效范围
        r, g, b = rgb_clamped[:, 0:1], rgb_clamped[:, 1:2], rgb_clamped[:, 2:3]

        # 2. 植物健康度通道（P通道）- 突出健康绿色区域
        sum_rgb = r + g + b + self.eps  # 防止除零
        green_norm = g / sum_rgb
        green_norm = torch.clamp(green_norm, self.eps, 1.0 - self.eps)

        # 抑制红色干扰（土壤、枯叶等）
        exp_arg = torch.clamp(-self.green_sensitivity * r, -10.0, 10.0)
        P = green_norm * torch.exp(exp_arg)

        # 3. 病变指示通道（R通道）- 突出黄/褐变区域
        g_safe = g + self.eps  # 防止g为零
        necrotic_index = (r + b) / g_safe  # 红黄分量占比（病害特征）
        necrotic_index = torch.clamp(necrotic_index, 0.0, 5.0)

        # 增强病斑与健康区域的对比度
        sigmoid_arg = torch.clamp(self.necrotic_sensitivity * (r - g), -10.0, 10.0)
        R = necrotic_index * torch.sigmoid(sigmoid_arg)

        # 4. 复合特征通道（C通道）- 融合亮度与纹理信息
        # 基础亮度分量
        intensity = (0.299 * r + 0.587 * g + 0.114 * b)
        intensity_compressed = self._bio_inspired_mapping(intensity)

        # 病变纹理特征
        scale_analysis = self._multi_scale_analysis(g)
        texture_component = scale_analysis * self.lesion_strength
        texture_component = torch.clamp(texture_component, 0.0, 1.0)

        # 复合通道
        C = intensity_compressed + texture_component

        # 5. 最终处理与拼接
        # 确保无NaN/inf并限制范围
        P = torch.clamp(P, self.eps, 1.0 - self.eps)
        R = torch.clamp(R, self.eps, 1.0 - self.eps)
        C = torch.clamp(C, self.eps, 1.0 - self.eps)

        # 规范化到[-1, 1]（适合神经网络输入）
        P = (P - 0.5) * 2.0
        R = (R - 0.5) * 2.0
        C = (C - 0.5) * 2.0

        return torch.cat([P, R, C], dim=1)