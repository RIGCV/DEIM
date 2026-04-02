"""

python train.py -c configs/deim_dfine/deim_hgnetv2_n_disease.yml --use-amp --seed=0

nohup python train.py -c configs/deim_dfine/deim_hgnetv2_n_disease.yml --use-amp --seed=0



DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""
import os  # 导入操作系统接口模块，用于文件和目录操作
import sys  # 导入系统相关模块，用于修改Python路径等操作
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # 将当前文件父目录添加到系统路径中，以便导入上层模块

import argparse  # 导入命令行参数解析模块，用于处理命令行输入

from engine.misc import dist_utils  # 从engine.misc导入分布式工具模块，用于分布式训练相关功能
from engine.core import YAMLConfig, yaml_utils  # 从engine.core导入YAML配置类和yaml工具函数
from engine.solver import TASKS  # 从engine.solver导入任务字典，用于获取对应的训练任务

debug=False  # 设置调试标志为False，用于控制是否启用调试模式

if debug:  # 如果调试模式开启
    import torch  # 导入PyTorch模块
    def custom_repr(self):  # 定义自定义的Tensor表示函数
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'  # 返回包含张量形状的自定义字符串表示
    original_repr = torch.Tensor.__repr__  # 保存原始的Tensor表示方法
    torch.Tensor.__repr__ = custom_repr  # 将Tensor的表示方法替换为自定义方法

def main(args, ) -> None:  # 定义主函数，接收命令行参数，无返回值
    """main
    """
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)  # 初始化分布式训练环境，设置打印等级、打印方法和随机种子

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'  # 如果同时使用则抛出错误信息


    update_dict = yaml_utils.parse_cli(args.update)  # 解析命令行中传入的更新配置参数，返回字典格式
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', ] and v is not None})  # 排除'update'键和值为None的参数

    cfg = YAMLConfig(args.config, **update_dict)  # 根据配置文件路径和更新字典创建YAML配置对象

    if args.resume or args.tuning:  # 如果是恢复训练或微调模式
        if 'HGNetv2' in cfg.yaml_cfg:  # 如果配置中包含HGNetv2网络
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False  # 将预训练标志设置为False，避免加载预训练权重

    print('cfg: ', cfg.__dict__)  # 打印配置对象的属性字典，用于调试和确认配置

    solver = TASKS[cfg.yaml_cfg['task']](cfg)  # 根据配置中的任务类型创建对应的求解器对象

    if args.test_only:  # 如果只进行测试
        solver.val()  # 调用求解器的验证方法进行评估
    else:  # 否则进行训练
        solver.fit()  # 调用求解器的训练方法开始训练

    dist_utils.cleanup()  # 清理分布式训练环境，释放资源


if __name__ == '__main__':  # 如果当前脚本作为主程序运行

    parser = argparse.ArgumentParser()  # 创建命令行参数解析器

    # priority 0  # 优先级0的参数，主要配置参数
    parser.add_argument('-c', '--config', type=str, required=True)  # 添加配置文件路径参数，必需
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')  # 添加恢复训练参数，从检查点恢复
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')  # 添加微调参数，从检查点微调
    parser.add_argument('-d', '--device', type=str, help='device',)  # 添加设备参数，指定运行设备(cpu/cuda)
    parser.add_argument('--seed', type=int, help='exp reproducibility')  # 添加随机种子参数，确保实验可重复性
    parser.add_argument('--use-amp', action='store_true', help='auto mixed precision training')  # 添加混合精度训练标志，启用自动混合精度
    parser.add_argument('--output-dir', type=str, help='output directoy')  # 添加输出目录参数，指定结果保存路径
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')  # 添加日志目录参数，指定TensorBoard日志路径
    parser.add_argument('--test-only', action='store_true', default=False,)  # 添加仅测试标志，只进行验证不训练

    # priority 1  # 优先级1的参数，次要配置参数
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')  # 添加更新配置参数，用于覆盖yaml文件中的配置

    # env  # 环境相关参数
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')  # 添加打印方法参数，指定日志打印方式
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')  # 添加打印等级参数，指定哪个进程打印日志

    parser.add_argument('--local-rank', type=int, help='local rank id')  # 添加本地排名参数，用于分布式训练
    args = parser.parse_args()  # 解析命令行参数，存储到args对象中

    main(args)  # 调用主函数，传入解析后的参数