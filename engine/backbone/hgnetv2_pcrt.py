import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .common import FrozenBatchNorm2d
from ..core import register
import logging
# from .hvi import RGB_HVI
from .pcrt import PRCT
# Constants for initialization
kaiming_normal_ = nn.init.kaiming_normal_
zeros_ = nn.init.zeros_
ones_ = nn.init.ones_

__all__ = ['HGNetv2_pcrt']


class LearnableAffineBlock(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, x):
        return self.scale * x + self.bias


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        kernel_size,
        stride=1,
        groups=1,
        padding='',
        use_act=True,
        use_lab=False
    ):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        if padding == 'same':
            self.conv = nn.Sequential(
                nn.ZeroPad2d([0, 1, 0, 1]),
                nn.Conv2d(in_chs, out_chs, kernel_size, stride, groups=groups, bias=False)
            )
        else:
            self.conv = nn.Conv2d(
                in_chs, out_chs, kernel_size, stride,
                padding=(kernel_size - 1) // 2, groups=groups, bias=False
            )
        self.bn = nn.BatchNorm2d(out_chs)
        self.act = nn.ReLU() if self.use_act else nn.Identity()
        self.lab = LearnableAffineBlock() if (self.use_act and self.use_lab) else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.lab(x)
        return x


class LightConvBNAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, groups=1, use_lab=False):
        super().__init__()
        self.conv1 = ConvBNAct(in_chs, out_chs, kernel_size=1, use_act=False, use_lab=use_lab)
        self.conv2 = ConvBNAct(out_chs, out_chs, kernel_size=kernel_size, groups=out_chs, use_act=True, use_lab=use_lab)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class StemBlock(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False):
        super().__init__()
        self.stem1 = ConvBNAct(in_chs, mid_chs, kernel_size=3, stride=2, use_lab=use_lab)
        self.stem2a = ConvBNAct(mid_chs, mid_chs // 2, kernel_size=2, stride=1, use_lab=use_lab)
        self.stem2b = ConvBNAct(mid_chs // 2, mid_chs, kernel_size=2, stride=1, use_lab=use_lab)
        self.stem3 = ConvBNAct(mid_chs * 2, mid_chs, kernel_size=3, stride=2, use_lab=use_lab)
        self.stem4 = ConvBNAct(mid_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(self, x):
        x = self.stem1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x)
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class EseModule(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.conv = nn.Conv2d(chs, chs, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = x.mean((2, 3), keepdim=True)
        x = self.conv(x)
        x = self.sigmoid(x)
        return torch.mul(identity, x)
class HG_Block(nn.Module):
    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        layer_num,
        kernel_size=3,
        residual=False,
        light_block=False,
        use_lab=False,
        agg='ese',
        drop_path=0.,
    ):
        super().__init__()
        self.residual = residual

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            if light_block:
                self.layers.append(
                    LightConvBNAct(in_chs if i == 0 else mid_chs, mid_chs, kernel_size=kernel_size, use_lab=use_lab)
                )
            else:
                self.layers.append(
                    ConvBNAct(in_chs if i == 0 else mid_chs, mid_chs, kernel_size=kernel_size, stride=1, use_lab=use_lab)
                )

        total_chs = in_chs + layer_num * mid_chs
        if agg == 'se':
            aggregation_squeeze_conv = ConvBNAct(total_chs, out_chs // 2, kernel_size=1, stride=1, use_lab=use_lab)
            aggregation_excitation_conv = ConvBNAct(out_chs // 2, out_chs, kernel_size=1, stride=1, use_lab=use_lab)
            self.aggregation = nn.Sequential(aggregation_squeeze_conv, aggregation_excitation_conv)
        else:
            aggregation_conv = ConvBNAct(total_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab)
            att = EseModule(out_chs)
            self.aggregation = nn.Sequential(aggregation_conv, att)

        self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        identity = x
        outputs = [x]
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        x = torch.cat(outputs, dim=1)
        x = self.aggregation(x)
        if self.residual:
            x = self.drop_path(x) + identity
        return x


class HG_Stage(nn.Module):
    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        block_num,
        layer_num,
        downsample=True,
        light_block=False,
        kernel_size=3,
        use_lab=False,
        agg='se',
        drop_path=0.,
    ):
        super().__init__()
        self.downsample = ConvBNAct(
            in_chs, in_chs, kernel_size=3, stride=2, groups=in_chs, use_act=False, use_lab=use_lab
        ) if downsample else nn.Identity()

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block(
                    in_chs if i == 0 else out_chs,
                    mid_chs,
                    out_chs,
                    layer_num,
                    residual=False if i == 0 else True,
                    kernel_size=kernel_size,
                    light_block=light_block,
                    use_lab=use_lab,
                    agg=agg,
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                )
            )
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class FusionBlock(nn.Module):
    """简单可靠的中层融合：concat -> 1x1 Conv -> BN -> ReLU"""
    def __init__(self, in_ch_each):
        super().__init__()
        self.conv = nn.Conv2d(in_ch_each * 2, in_ch_each, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_ch_each)
        self.act = nn.ReLU(inplace=True)

    def forward(self, frgb, fhvi):
        x = torch.cat([frgb, fhvi], dim=1)
        x = self.act(self.bn(self.conv(x)))
        return x


@register()
class HGNetv2_pcrt(nn.Module):
    """
    双分支：RGB 与 由 RGB_HVI 生成的 HVI
    在 stage2 / stage3 的分辨率处做特征交互融合
    """
    arch_configs = {
        'B0': {
            'stem_channels': [3, 16, 16],
            'stage_config': {
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True,  False, 3, 3],
                "stage3": [256, 64, 512, 2, True,  True,  5, 3],
                "stage4": [512, 128,1024, 1, True,  True,  5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        },
        'B1': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                "stage1": [32, 32, 64, 1, False, False, 3, 3],
                "stage2": [64, 48, 256, 1, True,  False, 3, 3],
                "stage3": [256, 96, 512, 2, True,  True,  5, 3],
                "stage4": [512, 192,1024, 1, True,  True,  5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B1_stage1.pth'
        },
        'B2': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                "stage1": [32, 32, 96, 1, False, False, 3, 4],
                "stage2": [96, 64, 384, 1, True,  False, 3, 4],
                "stage3": [384,128, 768, 3, True,  True,  5, 4],
                "stage4": [768,256,1536, 1, True,  True,  5, 4],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B2_stage1.pth'
        },
        'B3': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                "stage1": [32, 32, 128, 1, False, False, 3, 5],
                "stage2": [128,64, 512, 1, True,  False, 3, 5],
                "stage3": [512,128,1024, 3, True,  True,  5, 5],
                "stage4": [1024,256,2048, 1, True,  True,  5, 5],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B3_stage1.pth'
        },
        'B4': {
            'stem_channels': [3, 32, 48],
            'stage_config': {
                "stage1": [48, 48, 128, 1, False, False, 3, 6],
                "stage2": [128,96, 512, 1, True,  False, 3, 6],
                "stage3": [512,192,1024, 3, True,  True,  5, 6],
                "stage4": [1024,384,2048, 1, True,  True,  5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth'
        },
        'B5': {
            'stem_channels': [3, 32, 64],
            'stage_config': {
                "stage1": [64, 64, 128, 1, False, False, 3, 6],
                "stage2": [128,128, 512, 2, True,  False, 3, 6],
                "stage3": [512,256,1024, 5, True,  True,  5, 6],
                "stage4": [1024,512,2048, 2, True,  True,  5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B5_stage1.pth'
        },
        'B6': {
            'stem_channels': [3, 48, 96],
            'stage_config': {
                "stage1": [96, 96, 192, 2, False, False, 3, 6],
                "stage2": [192,192, 512, 3, True,  False, 3, 6],
                "stage3": [512,384,1024, 6, True,  True,  5, 6],
                "stage4": [1024,768,2048, 3, True,  True,  5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B6_stage1.pth'
        },
    }
    def __init__(
            self,
            name,
            use_lab=False,
            return_idx=[1, 2, 3],
            freeze_stem_only=True,
            freeze_at=0,
            freeze_norm=True,
            pretrained=True,
            local_model_dir='weight/hgnetv2/'
        ):
            super().__init__()
            self.use_lab = use_lab
            self.return_idx = return_idx

            # 伪模态生成器（RGB -> HVI 3ch）
            self.rgb2hvi = PRCT()

            stem_channels = self.arch_configs[name]['stem_channels']
            stage_config = self.arch_configs[name]['stage_config']
            download_url = self.arch_configs[name]['url']

            self._out_strides = [4, 8, 16, 32]
            self._out_channels = [stage_config[k][2] for k in stage_config]

            # === 双分支 stem（参数不共享，更稳） ===
            self.stem_rgb = StemBlock(
                in_chs=stem_channels[0], mid_chs=stem_channels[1], out_chs=stem_channels[2], use_lab=use_lab
            )
            self.stem_hvi = StemBlock(
                in_chs=3, mid_chs=stem_channels[1], out_chs=stem_channels[2], use_lab=use_lab
            )

            # === 双分支 stages（参数不共享）===
            self.stages_rgb = nn.ModuleList()
            self.stages_hvi = nn.ModuleList()
            for i, k in enumerate(stage_config):
                in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num = stage_config[k]
                self.stages_rgb.append(
                    HG_Stage(in_channels, mid_channels, out_channels, block_num, layer_num, downsample, light_block, kernel_size, use_lab)
                )
                self.stages_hvi.append(
                    HG_Stage(in_channels, mid_channels, out_channels, block_num, layer_num, downsample, light_block, kernel_size, use_lab)
                )

            # === 融合模块：在 stage2 / stage3 处交互 ===
            # 对齐 channels：stage2 的 in/out 通道可从配置读出
            # stage2 输出通道：
            stage_names = list(stage_config.keys())
            ch_stage2_out = stage_config[stage_names[1]][2]
            ch_stage3_out = stage_config[stage_names[2]][2]
            self.fuse2 = FusionBlock(in_ch_each=ch_stage2_out)
            self.fuse3 = FusionBlock(in_ch_each=ch_stage3_out)

            # === 冻结、归一化冻结 ===
            if freeze_at >= 0:
                self._freeze_parameters(self.stem_rgb)
                self._freeze_parameters(self.stem_hvi)
                if not freeze_stem_only:
                    for i in range(min(freeze_at + 1, len(self.stages_rgb))):
                        self._freeze_parameters(self.stages_rgb[i])
                        self._freeze_parameters(self.stages_hvi[i])

            if freeze_norm:
                self._freeze_norm(self)

            # === 预训练权重：仅加载与原版同名的键，其余跳过（strict=False）===
            if pretrained:
                RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
                try:
                    model_path = os.path.join(local_model_dir, f'PPHGNetV2_{name}_stage1.pth')
                    if os.path.exists(model_path):
                        state = torch.load(model_path, map_location='cpu')
                        print(f"Loaded stage1 {name} HGNetV2 from local file.")
                    else:
                        if torch.distributed.is_available() and torch.distributed.is_initialized():
                            rank0 = (torch.distributed.get_rank() == 0)
                        else:
                            rank0 = True
                        if rank0:
                            print(GREEN + "If the pretrained HGNetV2 can't be downloaded automatically. Please check your network connection." + RESET)
                            print(GREEN + "Please check your network connection. Or download the model manually from " + RESET + f"{download_url}" + GREEN + " to " + RESET + f"{local_model_dir}." + RESET)
                            state = torch.hub.load_state_dict_from_url(download_url, map_location='cpu', model_dir=local_model_dir)
                            if torch.distributed.is_available() and torch.distributed.is_initialized():
                                torch.distributed.barrier()
                        else:
                            if torch.distributed.is_available() and torch.distributed.is_initialized():
                                torch.distributed.barrier()
                            state = torch.load(model_path, map_location='cpu')

                        print(f"Loaded stage1 {name} HGNetV2 from URL.")

                    # 仅把原版 stem & stages 的权重加载到 RGB 分支；HVI 分支随机初始化
                    missing, unexpected = self._load_partial_pretrain(state_dict=state)
                    if len(unexpected) > 0:
                        print(RED + f"Unexpected keys in state_dict (ignored): {unexpected}" + RESET)
                    if len(missing) > 0:
                        print(RED + f"Missing keys not loaded (new modules or HVI branch): {missing}" + RESET)

                except (Exception, KeyboardInterrupt) as e:
                    print(f"{str(e)}")
                    logging.error(RED + "CRITICAL WARNING: Failed to load pretrained HGNetV2 model" + RESET)
                    logging.error(GREEN + "Please check your network connection. Or download the model manually from " \
                                + RESET + f"{download_url}" + GREEN + " to " + RESET + f"{local_model_dir}." + RESET)
                    # 不直接 exit()，避免影响训练流程
                    # exit()

    # --------- utils ---------
    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _load_partial_pretrain(self, state_dict):
        """
        把原版 HGNetV2 的权重尽可能加载到 RGB 分支（键名匹配），其余跳过。
        """
        # 构建一个“镜像”state_dict：把原模型里 stem -> stem_rgb, stages -> stages_rgb
        new_state = {}
        for k, v in state_dict.items():
            nk = k
            if k.startswith('stem.'):
                nk = 'stem_rgb.' + k[len('stem.'):]
            elif k.startswith('stages.'):
                nk = 'stages_rgb.' + k[len('stages.'):]
            new_state[nk] = v

        missing, unexpected = self.load_state_dict(new_state, strict=False)
        return missing, unexpected

    # --------- forward ---------
    def forward(self, x):
        """
        x: RGB, shape [B,3,H,W]
        流程：
          1) 生成 HVI（3ch），与 RGB 分别过自己的 stem
          2) 两支并行依次过各 stage
          3) 在 stage2 / stage3 的输出处做交互融合（更新两支为相同的 fused）
          4) 返回按照 return_idx 的 fused 特征
        """
        # 1) RGB -> HVI 模态（直接使用 HVI 三通道，不再还原）
        # with torch.no_grad():
            # 如果你希望 HVI 也可训练地参与梯度，这里去掉 no_grad()
        hvi = self.rgb2hvi.forward_rgb_to_prct(x)  # [B,3,H,W]

        # 2) 两支各自 stem
        xr = self.stem_rgb(x)   # [B, Cstem, H/4, W/4]
        xh = self.stem_hvi(hvi) # [B, Cstem, H/4, W/4]

        outs = []
        # 3) 逐 stage 前向 & 中层融合
        for idx, (stage_r, stage_h) in enumerate(zip(self.stages_rgb, self.stages_hvi)):
            xr = stage_r(xr)
            xh = stage_h(xh)

            # 在 stage2 / stage3 做交互（注意：idx从0开始 -> 1是stage2，2是stage3）
            if idx == 1:  # stage2
                fused = self.fuse2(xr, xh)
                xr = fused
                xh = fused
            if idx == 2:  # stage3
                fused = self.fuse3(xr, xh)
                xr = fused
                xh = fused

            # 收集输出（返回 fused）
            if idx in self.return_idx:
                # 两支已被同步为 fused，取任意一支即可
                outs.append(xr)

        return outs