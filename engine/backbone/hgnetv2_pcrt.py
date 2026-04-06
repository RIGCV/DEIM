import torch                      # 导入PyTorch主库
import torch.nn as nn             # 导入神经网络模块
import torch.nn.functional as F   # 导入函数式接口
import os                         # 导入操作系统接口
from .common import FrozenBatchNorm2d   # 导入冻结批归一化类
from ..core import register       # 导入注册装饰器
import logging                    # 导入日志模块
# from .hvi import RGB_HVI        # 已注释：原本的HVI生成模块
from .pcrt import PRCT            # 导入PRCT类，用于RGB到PRCT（HVI）转换

# Constants for initialization    # 初始化常量
kaiming_normal_ = nn.init.kaiming_normal_   # Kaiming正态初始化别名
zeros_ = nn.init.zeros_           # 全零初始化别名
ones_ = nn.init.ones_             # 全1初始化别名

__all__ = ['HGNetv2_pcrt']        # 模块公开接口列表


class LearnableAffineBlock(nn.Module):   # 【可学习仿射块】
    def __init__(self, scale_value=1.0, bias_value=0.0):   # 默认缩放1偏置0
        super().__init__()        # 调用父类初始化
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)   # 可学习缩放参数
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)     # 可学习偏置参数

    def forward(self, x):         # 前向传播,input x
        return self.scale * x + self.bias   # 输出 = 缩放×输入 + 偏置


class ConvBNAct(nn.Module):       # 【卷积+批归一化BN+激活函数模块】
    def __init__(self, in_chs, out_chs, kernel_size, stride=1, groups=1, padding='',
                 use_act=True, use_lab=False):   # 参数列表，use_act：是否在BN后使用ReLU，use_lab：是否在激活后使用可学习仿射块（仅在use_act=True时有效）
        super().__init__()        # 父类初始化
        self.use_act = use_act    # 是否使用激活函数
        self.use_lab = use_lab    # 是否使用可学习仿射块
        if padding == 'same':     # 如果需要'same'填充
            self.conv = nn.Sequential(          # 顺序容器
                nn.ZeroPad2d([0, 1, 0, 1]),     # 右下角填充1行/列
                nn.Conv2d(in_chs, out_chs, kernel_size, stride, groups=groups, bias=False)   # 无偏置卷积
            )
        else:                     # 默认对称填充
            self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride,
                                  padding=(kernel_size - 1) // 2, groups=groups, bias=False)   # 保持尺寸
        self.bn = nn.BatchNorm2d(out_chs)       # 批归一化层
        self.act = nn.ReLU() if self.use_act else nn.Identity()   # 激活或恒等
        self.lab = LearnableAffineBlock() if (self.use_act and self.use_lab) else nn.Identity()   # LAB或恒等

    def forward(self, x):         # 前向传播
        x = self.conv(x)          # 卷积操作
        x = self.bn(x)            # 批归一化
        x = self.act(x)           # 激活函数
        x = self.lab(x)           # 可学习仿射（若启用）
        return x                  # 返回结果


class LightConvBNAct(nn.Module):  # 【轻量卷积块：1x1卷积 + 深度可分离卷积】
    def __init__(self, in_chs, out_chs, kernel_size, groups=1, use_lab=False):   # 初始化
        super().__init__()        # 父类初始化
        self.conv1 = ConvBNAct(in_chs, out_chs, kernel_size=1, use_act=False, use_lab=use_lab)   # 点卷积，无激活
        self.conv2 = ConvBNAct(out_chs, out_chs, kernel_size=kernel_size, groups=out_chs, use_act=True, use_lab=use_lab)   # 深度卷积

    def forward(self, x):         # 前向传播
        x = self.conv1(x)         # 点卷积升/降维
        x = self.conv2(x)         # 深度卷积提取空间特征
        return x                  # 返回结果


class StemBlock(nn.Module):       # 初始stem模块：下采样+特征组合
    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False):   # 初始化
        super().__init__()        # 父类初始化
        self.stem1 = ConvBNAct(in_chs, mid_chs, kernel_size=3, stride=2, use_lab=use_lab)   # 步长2下采样
        self.stem2a = ConvBNAct(mid_chs, mid_chs // 2, kernel_size=2, stride=1, use_lab=use_lab)   # 分支a：减半通道
        self.stem2b = ConvBNAct(mid_chs // 2, mid_chs, kernel_size=2, stride=1, use_lab=use_lab)   # 分支b：恢复通道
        self.stem3 = ConvBNAct(mid_chs * 2, mid_chs, kernel_size=3, stride=2, use_lab=use_lab)     # 下采样并降维
        self.stem4 = ConvBNAct(mid_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab)         # 1x1卷积调整通道
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)   # 最大池化，ceil模式

    def forward(self, x):         # 前向传播
        x = self.stem1(x)         # 第一次卷积+下采样
        x = F.pad(x, (0, 1, 0, 1))   # 右侧和下侧填充1行/列
        x2 = self.stem2a(x)       # 分支a处理
        x2 = F.pad(x2, (0, 1, 0, 1))   # 再次填充
        x2 = self.stem2b(x2)      # 分支b处理
        x1 = self.pool(x)         # 池化分支
        x = torch.cat([x1, x2], dim=1)   # 沿通道拼接
        x = self.stem3(x)         # 下采样卷积
        x = self.stem4(x)         # 最终通道调整
        return x                  # 返回结果


class EseModule(nn.Module):       # ESE通道注意力模块，ESE：Efficient Squeeze-Excitation，与标准SE不同之处在于：全局池化后直接使用1x1卷积（而非两个全连接层），计算量更小。
    def __init__(self, chs):      # 初始化
        super().__init__()        # 父类初始化
        self.conv = nn.Conv2d(chs, chs, kernel_size=1, stride=1, padding=0)   # 1x1卷积
        self.sigmoid = nn.Sigmoid()   # Sigmoid激活

    def forward(self, x):         # 前向传播
        identity = x              # 保存残差输入
        x = x.mean((2, 3), keepdim=True)   # 全局平均池化 → [B,C,1,1]
        x = self.conv(x)          # 1x1卷积学习权重
        x = self.sigmoid(x)       # Sigmoid得到0~1权重
        return torch.mul(identity, x)   # 注意力加权


class HG_Block(nn.Module):        # HGNet基本块：多分支叠加+聚合
    def __init__(self, in_chs, mid_chs, out_chs, layer_num, kernel_size=3, residual=False,
                 light_block=False, use_lab=False, agg='ese', drop_path=0.):   # 参数列表
        super().__init__()        # 父类初始化
        self.residual = residual  # 保存残差标志
        self.layers = nn.ModuleList()   # 模块列表容器
        for i in range(layer_num):   # 循环构建每一层
            if light_block:        # 如果使用轻量块
                self.layers.append(LightConvBNAct(in_chs if i == 0 else mid_chs, mid_chs,
                                                  kernel_size=kernel_size, use_lab=use_lab))   # 轻量卷积
            else:                  # 否则使用标准卷积块
                self.layers.append(ConvBNAct(in_chs if i == 0 else mid_chs, mid_chs,
                                             kernel_size=kernel_size, stride=1, use_lab=use_lab))   # 标准卷积
        total_chs = in_chs + layer_num * mid_chs   # 所有层输出通道总和（含输入）
        if agg == 'se':            # SE聚合方式
            aggregation_squeeze_conv = ConvBNAct(total_chs, out_chs // 2, kernel_size=1, stride=1, use_lab=use_lab)   # 降维
            aggregation_excitation_conv = ConvBNAct(out_chs // 2, out_chs, kernel_size=1, stride=1, use_lab=use_lab)   # 升维
            self.aggregation = nn.Sequential(aggregation_squeeze_conv, aggregation_excitation_conv)   # SE顺序容器
        else:                      # ESE聚合方式
            aggregation_conv = ConvBNAct(total_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab)   # 1x1卷积
            att = EseModule(out_chs)   # ESE注意力
            self.aggregation = nn.Sequential(aggregation_conv, att)   # 卷积+注意力
        self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity()   # DropPath或恒等

    def forward(self, x):         # 前向传播
        identity = x              # 保存残差输入
        outputs = [x]             # 输出列表，初始含输入
        for layer in self.layers:  # 逐层处理
            x = layer(x)          # 当前层输出
            outputs.append(x)     # 添加到列表
        x = torch.cat(outputs, dim=1)   # 沿通道拼接
        x = self.aggregation(x)   # 聚合处理
        if self.residual:         # 如果使用残差
            x = self.drop_path(x) + identity   # 随机丢弃路径后加残差
        return x                  # 返回结果


class HG_Stage(nn.Module):        # HGNet阶段：下采样 + 多个HG_Block
    def __init__(self, in_chs, mid_chs, out_chs, block_num, layer_num, downsample=True,
                 light_block=False, kernel_size=3, use_lab=False, agg='se', drop_path=0.):   # 参数列表
        super().__init__()        # 父类初始化
        self.downsample = ConvBNAct(in_chs, in_chs, kernel_size=3, stride=2, groups=in_chs,
                                    use_act=False, use_lab=use_lab) if downsample else nn.Identity()   # 下采样或恒等
        blocks_list = []          # 块列表
        for i in range(block_num):   # 循环构建每个HG_Block
            blocks_list.append(HG_Block(
                in_chs if i == 0 else out_chs,   # 第一个块输入为in_chs，之后为out_chs
                mid_chs, out_chs, layer_num,
                residual=False if i == 0 else True,   # 第一个块无残差，之后有残差
                kernel_size=kernel_size,
                light_block=light_block,
                use_lab=use_lab,
                agg=agg,
                drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path   # 支持逐块丢弃率
            ))
        self.blocks = nn.Sequential(*blocks_list)   # 顺序容器包装

    def forward(self, x):         # 前向传播
        x = self.downsample(x)    # 下采样（若启用）
        x = self.blocks(x)        # 通过所有块
        return x                  # 返回结果


class FusionBlock(nn.Module):     # 简单融合模块：concat + 1x1卷积 + BN + ReLU
    """简单可靠的中层融合：concat -> 1x1 Conv -> BN -> ReLU"""   # 文档字符串

    def __init__(self, in_ch_each):   # 初始化
        super().__init__()        # 父类初始化
        self.conv = nn.Conv2d(in_ch_each * 2, in_ch_each, kernel_size=1, bias=False)   # 1x1卷积，双倍输入
        self.bn = nn.BatchNorm2d(in_ch_each)   # 批归一化
        self.act = nn.ReLU(inplace=True)       # 就地ReLU

    def forward(self, frgb, fhvi):   # 前向传播
        x = torch.cat([frgb, fhvi], dim=1)   # 沿通道拼接
        x = self.act(self.bn(self.conv(x)))  # 卷积->BN->ReLU
        return x                  # 返回融合结果

class WeightedFeatureFusion(nn.Module):
    """
    可学习权重的特征加权求和融合
    fused = w1*feat1 + w2*feat2 + w3*feat3 + ...
    自动约束权重和为1，保证数值稳定
    """
    def __init__(self, num_features: 2):
        """
        Args:
            num_features: 要融合的特征数量，比如融合2个特征就填2
        """
        super().__init__()
        # 初始化可学习权重，初始值相等
        self.weights = nn.Parameter(torch.ones(num_features))
        
    def forward(self, *features):
        """
        Args:
            *features: 传入任意多个特征图，必须 shape 完全一致
        Returns:
            fused_feature: 加权融合后的特征图
        """
        # 权重归一化（保证和为1，训练更稳定）
        normalized_weights = F.softmax(self.weights, dim=0)
        
        # 加权求和
        fused_feature = 0.0
        for i, feat in enumerate(features):
            fused_feature += normalized_weights[i] * feat
            
        return fused_feature
#class FusionBlock(nn.Module):  
#作用：将两个分支（RGB和HVI）相同分辨率的特征图拼接后，通过1x1卷积降维回原始通道数，
# 然后BN+ReLU。输出与输入的单分支具有相同形状，因此可以直接替换两支的特征。

@register()                       # 注册装饰器
class HGNetv2_pcrt(nn.Module):    # 双分支HGNetv2，支持RGB和PRCT(HVI)模态融合
    """
    双分支：RGB 与 由 RGB_HVI 生成的 HVI
    在 stage2 / stage3 的分辨率处做特征交互融合
    """
    arch_configs = {              # 架构配置字典
        'B0': {                   # B0配置
            'stem_channels': [3, 16, 16],
            'stage_config': {
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 2, True, True, 5, 3],
                "stage4": [512, 128, 1024, 1, True, True, 5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        },
        'B1': {                   # B1配置
            'stem_channels': [3, 24, 32],
            'stage_config': {
                "stage1": [32, 32, 64, 1, False, False, 3, 3],
                "stage2": [64, 48, 256, 1, True, False, 3, 3],
                "stage3": [256, 96, 512, 2, True, True, 5, 3],
                "stage4": [512, 192, 1024, 1, True, True, 5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B1_stage1.pth'
        },
        'B2': {                   # B2配置
            'stem_channels': [3, 24, 32],
            'stage_config': {
                "stage1": [32, 32, 96, 1, False, False, 3, 4],
                "stage2": [96, 64, 384, 1, True, False, 3, 4],
                "stage3": [384, 128, 768, 3, True, True, 5, 4],
                "stage4": [768, 256, 1536, 1, True, True, 5, 4],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B2_stage1.pth'
        },
        'B3': {                   # B3配置
            'stem_channels': [3, 24, 32],
            'stage_config': {
                "stage1": [32, 32, 128, 1, False, False, 3, 5],
                "stage2": [128, 64, 512, 1, True, False, 3, 5],
                "stage3": [512, 128, 1024, 3, True, True, 5, 5],
                "stage4": [1024, 256, 2048, 1, True, True, 5, 5],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B3_stage1.pth'
        },
        'B4': {                   # B4配置
            'stem_channels': [3, 32, 48],
            'stage_config': {
                "stage1": [48, 48, 128, 1, False, False, 3, 6],
                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                "stage3": [512, 192, 1024, 3, True, True, 5, 6],
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth'
        },
        'B5': {                   # B5配置
            'stem_channels': [3, 32, 64],
            'stage_config': {
                "stage1": [64, 64, 128, 1, False, False, 3, 6],
                "stage2": [128, 128, 512, 2, True, False, 3, 6],
                "stage3": [512, 256, 1024, 5, True, True, 5, 6],
                "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B5_stage1.pth'
        },
        'B6': {                   # B6配置
            'stem_channels': [3, 48, 96],
            'stage_config': {
                "stage1": [96, 96, 192, 2, False, False, 3, 6],
                "stage2": [192, 192, 512, 3, True, False, 3, 6],
                "stage3": [512, 384, 1024, 6, True, True, 5, 6],
                "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B6_stage1.pth'
        },
    }

    def __init__(self, name, use_lab=False, return_idx=[1, 2, 3], freeze_stem_only=True,
                 freeze_at=0, freeze_norm=True, pretrained=True, local_model_dir='weight/hgnetv2/'):   # 初始化参数   local_model_dir：本地存储预训练权重的目录。
        super().__init__()        # 父类初始化
        self.use_lab = use_lab    # 保存use_lab标志
        self.return_idx = return_idx   # 要返回的阶段索引

        # 伪模态生成器（RGB -> HVI 3ch）
        self.rgb2hvi = PRCT()     # 实例化PRCT转换器

        stem_channels = self.arch_configs[name]['stem_channels']   # 获取stem通道配置
        stage_config = self.arch_configs[name]['stage_config']
        download_url = self.arch_configs[name]['url']              # 获取预训练URL

        self._out_strides = [4, 8, 16, 32]   # 各阶段下采样倍数
        self._out_channels = [stage_config[k][2] for k in stage_config] # 各阶段输出通道

        # === 双分支 stem（参数不共享） ===
        self.stem_rgb = StemBlock(in_chs=stem_channels[0], mid_chs=stem_channels[1],
                                  out_chs=stem_channels[2], use_lab=use_lab)   # RGB分支stem
        self.stem_hvi = StemBlock(in_chs=3, mid_chs=stem_channels[1],
                                  out_chs=stem_channels[2], use_lab=use_lab)   # HVI分支stem

        # === 双分支 stages（参数不共享） ===
        self.stages_rgb = nn.ModuleList()   # RGB阶段容器
        self.stages_hvi = nn.ModuleList()   # HVI阶段容器
        stage_names = list(stage_config.keys())
        for i, k in enumerate(stage_names):
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num = stage_config[k]

            # RGB 分支保留所有 stage
            self.stages_rgb.append(
                HG_Stage(in_channels, mid_channels, out_channels, block_num, layer_num, downsample, light_block,
                         kernel_size, use_lab)
            )

            # HVI 分支只保留前三个 stage
            if i < 3:
                self.stages_hvi.append(
                    HG_Stage(in_channels, mid_channels, out_channels, block_num, layer_num, downsample, light_block,
                             kernel_size, use_lab)
                )

        # === 融合模块：stage2 / stage3 交互 ===
        stage_names = list(stage_config.keys())   # 阶段名称列表
        ch_stage2_out = stage_config[stage_names[1]][2]   # stage2输出通道
        ch_stage3_out = stage_config[stage_names[2]][2]   # stage3输出通道
        # self.fuse2 = FusionBlock(in_ch_each=ch_stage2_out)   # stage2融合模块
        # self.fuse3 = FusionBlock(in_ch_each=ch_stage3_out)   # stage3融合模块
        self.fuse2 = WeightedFeatureFusion(2)
        self.fuse3 = WeightedFeatureFusion(2)

        # === 冻结参数 ===
        if freeze_at >= 0:        # 需要冻结
            self._freeze_parameters(self.stem_rgb)   # 冻结RGB stem
            self._freeze_parameters(self.stem_hvi)   # 冻结HVI stem
            if not freeze_stem_only:   # 如果不是仅冻结stem
                for i in range(min(freeze_at + 1, len(self.stages_rgb))):   # 前freeze_at+1个阶段
                    self._freeze_parameters(self.stages_rgb[i])   # 冻结RGB阶段
                    self._freeze_parameters(self.stages_hvi[i])   # 冻结HVI阶段

        if freeze_norm:            # 需要冻结归一化层
            self._freeze_norm(self)   # 递归替换BN为冻结版本

        # === 加载预训练权重（仅RGB分支） ===
        if pretrained:             # 需要预训练
            RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"   # 终端颜色
            try:                   # 尝试加载
                model_path = os.path.join(local_model_dir, f'PPHGNetV2_{name}_stage1.pth')   # 本地路径
                if os.path.exists(model_path):   # 本地存在
                    state = torch.load(model_path, map_location='cpu')   # 加载状态字典
                    print(f"Loaded stage1 {name} HGNetV2 from local file.")   # 打印信息
                else:              # 需要下载
                    if torch.distributed.is_available() and torch.distributed.is_initialized():   # 分布式
                        rank0 = (torch.distributed.get_rank() == 0)   # 是否rank0
                    else:
                        rank0 = True   # 非分布式视为rank0
                    if rank0:        # rank0负责下载
                        print(GREEN + "If the pretrained HGNetV2 can't be downloaded automatically. Please check your network connection." + RESET)   # 提示
                        print(GREEN + "Please check your network connection. Or download the model manually from " + RESET + f"{download_url}" + GREEN + " to " + RESET + f"{local_model_dir}." + RESET)   # 指导
                        state = torch.hub.load_state_dict_from_url(download_url, map_location='cpu', model_dir=local_model_dir)   # 下载
                        if torch.distributed.is_available() and torch.distributed.is_initialized():   # 同步
                            torch.distributed.barrier()   # 等待所有进程
                    else:            # 非rank0等待下载完成
                        if torch.distributed.is_available() and torch.distributed.is_initialized():   # 同步
                            torch.distributed.barrier()   # 等待rank0
                        state = torch.load(model_path, map_location='cpu')   # 加载本地文件
                    print(f"Loaded stage1 {name} HGNetV2 from URL.")   # 打印信息
                # 部分加载：stem→stem_rgb, stages→stages_rgb
                missing, unexpected = self._load_partial_pretrain(state_dict=state)   # 非严格加载
                if len(unexpected) > 0:   # 有意外键
                    print(RED + f"Unexpected keys in state_dict (ignored): {unexpected}" + RESET)   # 打印警告
                if len(missing) > 0:      # 有缺失键
                    print(RED + f"Missing keys not loaded (new modules or HVI branch): {missing}" + RESET)   # 打印缺失
            except (Exception, KeyboardInterrupt) as e:   # 异常处理
                print(f"{str(e)}")        # 打印异常
                logging.error(RED + "CRITICAL WARNING: Failed to load pretrained HGNetV2 model" + RESET)   # 错误日志
                logging.error(GREEN + "Please check your network connection. Or download the model manually from "
                              + RESET + f"{download_url}" + GREEN + " to " + RESET + f"{local_model_dir}." + RESET)   # 指导
                # 不直接exit()，避免影响训练

    # --------- utils ---------
    def _freeze_norm(self, m: nn.Module):   # 递归冻结BN层
        if isinstance(m, nn.BatchNorm2d):   # 如果是BatchNorm2d
            m = FrozenBatchNorm2d(m.num_features)   # 替换为冻结版本
        else:                         # 否则遍历子模块
            for name, child in m.named_children():   # 遍历子模块
                _child = self._freeze_norm(child)    # 递归处理
                if _child is not child:   # 如果返回了新模块
                    setattr(m, name, _child)   # 替换
        return m                      # 返回模块

    def _freeze_parameters(self, m: nn.Module):   # 冻结模块所有参数
        for p in m.parameters():      # 遍历参数
            p.requires_grad = False   # 禁止梯度

    @torch.no_grad()                  # 不计算梯度
    def _load_partial_pretrain(self, state_dict):   # 部分加载预训练权重
        """把原版HGNetV2的权重加载到RGB分支，其余跳过"""
        new_state = {}                # 新状态字典
        for k, v in state_dict.items():   # 遍历原字典
            nk = k                    # 默认键名
            if k.startswith('stem.'):   # stem前缀
                nk = 'stem_rgb.' + k[len('stem.'):]   # 改为stem_rgb
            elif k.startswith('stages.'):   # stages前缀
                nk = 'stages_rgb.' + k[len('stages.'):]   # 改为stages_rgb
            new_state[nk] = v         # 存入新字典
        missing, unexpected = self.load_state_dict(new_state, strict=False)   # 非严格加载
        return missing, unexpected    # 返回缺失和多余键

    # --------- forward ---------
    def forward(self, x):             # 前向传播
        """
        x: RGB, shape [B,3,H,W]
        流程：
          1) 生成 HVI（3ch），与 RGB 分别过自己的 stem
          2) 两支并行依次过各 stage
          3) 在 stage2 / stage3 的输出处做交互融合（更新两支为相同的 fused）
          4) 返回按照 return_idx 的 fused 特征
        """
        # 1) RGB -> HVI 模态
        hvi = self.rgb2hvi.forward_rgb_to_prct(x)   # 生成PRCT(HVI)特征

        # 2) 两支各自 stem
        xr = self.stem_rgb(x)         # RGB分支stem → [B,Cstem,H/4,W/4]
        xh = self.stem_hvi(hvi)       # HVI分支stem → [B,Cstem,H/4,W/4]

        outs = []                     # 存储输出特征
        # 3) 逐stage前向 + 中层融合
             # 添加融合后的特征（两分支相同）
        for idx, stage_r in enumerate(self.stages_rgb):
            xr = stage_r(xr)

            # HVI 分支只遍历前3个 stage
            if idx < 3:
                xh = self.stages_hvi[idx](xh)
                if idx == 1:
                    fused = self.fuse2(xr, xh)
                    xr = fused
                    xh = fused
                if idx == 2:  # stage2
                    fused = self.fuse3(xr, xh)
                    xr = fused
                    xh = fused

            if idx in self.return_idx:
                outs.append(xr)
        return outs                   # 返回特征列表
