import sys
import math
import time
import os
from typing import Iterable
from datetime import datetime  # 新增：用于格式化输出系统时间，仅新增该行无其他新增

import torch
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler
from tqdm import tqdm  # 进度条可视化库

# 自定义模块导入
from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils

# ===================== 全局最优指标存储 - 核心功能 =====================
# 用于记录整个训练过程中，所有训练/评估指标的最优值 & 对应轮次，全局生效
BEST_METRICS = {
    "best_avg_loss": float('inf'),       # 最优平均总损失(越小越好)
    "best_avg_loss_epoch": -1,           # 最优损失对应的轮次
    "best_ap_all": 0.0,                  # 最优AP@[0.5:0.95]综合指标(越大越好)
    "best_ap_all_epoch": -1,             # 最优综合AP对应的轮次
    "best_ap_05": 0.0,                   # 最优AP@0.5主指标(越大越好)
    "best_ap_05_epoch": -1,              # 最优AP@0.5对应的轮次
    "best_ap_75": 0.0,                   # 最优AP@0.75高精度指标(越大越好)
    "best_ap_75_epoch": -1,              # 最优AP@0.75对应的轮次
}

def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    """
    单轮训练函数
    Args:
        self_lr_scheduler: 是否使用迭代式学习率调度
        lr_scheduler: 学习率调度器
        model: 待训练模型
        criterion: 损失函数
        data_loader: 训练数据加载器
        optimizer: 优化器
        device: 训练设备(GPU/CPU)
        epoch: 当前训练轮次
        max_norm: 梯度裁剪最大范数
        **kwargs: 额外参数(writer/ema/scaler等)
    Returns:
        dict: 当前轮次平均损失
    """
    # 模型切换为训练模式：开启梯度计算、BN层更新、Dropout生效等
    model.train()
    # 损失函数切换为训练模式：部分损失函数有训练/评估差异（如带正则的损失）
    criterion.train()
    
    # 从kwargs字典中获取打印频率，无则使用默认值100，安全取值不报错
    print_freq = kwargs.get('print_freq', 100)
    # 获取TensorBoard日志写入器，无则为None，不写入日志
    writer: SummaryWriter = kwargs.get('writer', None)
    # 获取EMA指数移动平均模型，无则为None，不使用EMA权重平滑
    ema: ModelEMA = kwargs.get('ema', None)
    # 获取混合精度训练的梯度缩放器，无则为None，使用普通精度训练
    scaler: GradScaler = kwargs.get('scaler', None)
    # 获取学习率预热调度器，无则为None，不使用学习率预热
    lr_warmup_scheduler: Warmup = kwargs.get('lr_warmup_scheduler', None)
    
    # 是否显示训练进度条，默认开启
    show_progress = kwargs.get('show_progress_bar', True)
    # 是否打印详细日志，默认关闭，精简输出
    verbose_logging = kwargs.get('verbose_logging', False)
    
    # 检测是否在nohup后台运行：stdout不是终端则判定为后台运行
    is_nohup = not sys.stdout.isatty()
    
    # 当前epoch的总迭代步数 = 数据集总样本数 / 批次大小
    total_iters = len(data_loader)

    # ========== 核心修改点1：训练轮次显示 +1 ，从001开始 ==========
    show_epoch = epoch + 1
    
    # ========== 训练轮次头部醒目分隔 - 保留原优化格式 ==========
    if dist_utils.is_main_process():
        print("\n" + "="*100)
        # ✅ 优化要求1：训练开始前 打印当前系统时间 - 新增核心代码
        current_sys_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{current_sys_time}🏆🏆第【{show_epoch:03d}】轮开始训练🏆🏆".center(80))
        print("="*100)
        print(f"📦 总迭代数: {total_iters}")
        print(f"📦 批次大小: {data_loader.batch_size}")
        print(f"📦 混合精度训练: {scaler is not None} ")
        print(f"📦 当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"📦 训练模式: {'后台nohup' if is_nohup else '前台交互式'}")
        print("="*100 + "")
    
    # 初始化进度条 - nohup后台模式不启用tqdm，避免日志刷屏，前台模式正常显示
    pbar = None
    if show_progress and dist_utils.is_main_process() and not is_nohup:
        pbar = tqdm(total=total_iters, 
                   desc=f'Epoch {show_epoch:03d}',
                   ncols=110,
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
                   colour='green')
    
    # 初始化训练统计核心变量
    total_loss = 0.0          # 累计当前epoch的总损失值
    start_time = time.time()  # 当前epoch的训练开始时间，用于计算耗时/速度
    last_print_time = start_time # 上一次打印日志的时间，用于控制日志刷新频率

    # 遍历数据集加载器，逐批次训练，i=迭代步数，samples=图像数据，targets=标注数据
    for i, (samples, targets) in enumerate(data_loader):
        # 将图像数据转移到指定设备(GPU/CPU) + non_blocking=True加速传输
        samples = samples.to(device, non_blocking=True)
        # 将标注数据的每一个字段都转移到指定设备，non_blocking加速数据传输
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        # 计算全局迭代步数 = 当前轮次*本轮总迭代数 + 当前迭代数，用于TensorBoard日志/学习率调度
        global_step = epoch * len(data_loader) + i
        # 构建训练元信息，传给损失函数，部分损失需要轮次/步数信息做动态调整
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        # ========== 混合精度训练逻辑 ==========
        if scaler is not None:
            # 开启自动混合精度，自动将张量转为半精度，加速训练+减少显存占用
            with torch.autocast(device_type=str(device), cache_enabled=True):
                # 模型前向推理，传入图像和标注，输出预测结果
                outputs = model(samples, targets=targets)

            # 检测预测框是否出现NaN/无穷大值，出现则保存权重并终止训练，避免训练崩溃
            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                print(f"❌ 预测框出现NaN/Inf，保存异常权重并退出")
                # 清理模型权重的module前缀，适配单机单卡/多卡
                state = {k.replace('module.', ''): v for k, v in model.state_dict().items()}
                dist_utils.save_on_master({'model': state}, "./NaN_error.pth")
                if pbar:
                    pbar.close() # 关闭进度条
                sys.exit(1) # 终止程序

            # 关闭自动混合精度计算损失，避免损失值精度不足导致梯度异常
            with torch.autocast(device_type=str(device), enabled=False):
                # 计算损失字典，包含各类分项损失
                loss_dict = criterion(outputs, targets,** metas)

            # 计算总损失 = 所有分项损失求和
            loss = sum(loss_dict.values())
            # 梯度缩放，放大损失值的梯度，避免半精度梯度下溢
            scaler.scale(loss).backward()

            # 梯度裁剪，防止梯度爆炸，max_norm>0时生效
            if max_norm > 0:
                scaler.unscale_(optimizer) # 对优化器梯度反缩放，恢复真实梯度值
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm) # 按L2范数裁剪梯度

            # 优化器步进更新权重，scaler自动处理梯度缩放逻辑
            scaler.step(optimizer)
            # 更新梯度缩放器的缩放系数，动态调整缩放比例
            scaler.update()
            # 清空优化器梯度，避免梯度累加
            optimizer.zero_grad(set_to_none=True) # set_to_none=True更节省显存

        # ========== 普通精度训练逻辑 ==========
        else:
            # 模型前向推理
            outputs = model(samples, targets=targets)
            # 计算损失字典
            loss_dict = criterion(outputs, targets, **metas)
            # 计算总损失
            loss: torch.Tensor = sum(loss_dict.values())
            # 清空优化器梯度
            optimizer.zero_grad(set_to_none=True)
            # 反向传播计算梯度
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            # 优化器步进更新权重
            optimizer.step()

        # EMA权重更新：如果启用EMA，则用当前模型权重更新EMA模型
        if ema is not None:
            ema.update(model)

        # ========== 学习率调度逻辑 ==========
        if self_lr_scheduler:
            # 迭代式学习率调度，按全局步数更新学习率
            optimizer = lr_scheduler.step(global_step, optimizer)
        else:
            # 预热学习率调度，只在预热阶段生效
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        # 分布式训练：聚合多卡的损失值，取均值，保证多卡日志一致
        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        # 计算聚合后的总损失值
        loss_value = sum(loss_dict_reduced.values())
        # ✅ 修复BUG：张量取值标准化，先detach再item，避免计算图残留+警告，兼容所有场景
        loss_value_scalar = loss_value.detach().item() if torch.is_tensor(loss_value) else float(loss_value)

        # 检测损失值是否为无穷大/NaN，出现则终止训练
        if not math.isfinite(loss_value_scalar):
            if pbar:
                pbar.close()
            print(f"\n❌ 致命错误: 损失值异常 -> {loss_value_scalar}")
            print(f"异常损失明细: {loss_dict_reduced}")
            sys.exit(1)
        
        # 累计当前epoch的总损失，用于计算平均损失
        total_loss += loss_value_scalar
        
        # ========== 过滤无用辅助损失，只保留训练核心损失组件 ==========
        filtered_loss_dict = {}
        for k, v in loss_dict_reduced.items():
            # 剔除所有辅助损失/去噪损失/预热损失，只保留核心训练损失，精简显示
            if not any(kw in k for kw in ['aux_', 'dn_', '_pre', '_enc', '_aux', '_dn']):
                filtered_loss_dict[k] = v.detach().item() if torch.is_tensor(v) else float(v)
        
        # ========== 进度条/日志输出优化 - 精简+重点突出 ==========
        current_time = time.time()
        if pbar:  # 前台进度条模式
            # 【批次输出频率】前台进度条-批次日志更新规则：满足任一条件即更新
            # 1. 距离上次打印超1.5秒  2. 每50个批次强制更新  3. 最后一个批次强制更新
            if current_time - last_print_time > 1.5 or i % 50 == 0 or i == total_iters - 1:
                current_lr = optimizer.param_groups[0]["lr"] # 获取当前学习率
                # ✅ 修复BUG：显存计算兼容CPU+GPU，无cuda时不报错，且计算当前显存占用而非峰值
                if torch.cuda.is_available() and device.type == 'cuda':
                    mem_used = torch.cuda.memory_allocated(device) / 1024**3
                else:
                    mem_used = 0.0
                # 计算训练速度 = 已迭代数 / 已用时间
                speed = (i + 1) / (current_time - start_time) if (current_time - start_time) > 0 else 0
                
                # 进度条后缀：只显示核心指标，格式统一整洁，修复loss重复显示bug
                postfix = f"总损失:{loss_value_scalar:.3f} | LR:{current_lr:.6f} | 显存:{mem_used:.1f}G | {speed:.1f}it/s"
                # 追加核心损失组件，不冗余
                for k, v in filtered_loss_dict.items():
                    postfix += f" | {k}:{v:.3f}"
                pbar.set_postfix_str(postfix)
                last_print_time = current_time
            pbar.update(1) # 更新进度条步数
        
        elif is_nohup and dist_utils.is_main_process():  # nohup后台日志模式，极简无冗余
            # 【批次输出频率】后台nohup模式-批次日志打印规则：
            # 1. 每500个批次打印一次  2. 最后一个批次强制打印  无其他刷新规则
            if i % 500 == 0 or i == total_iters - 1:
                current_lr = optimizer.param_groups[0]["lr"]
                speed = (i + 1) / (current_time - start_time) if (current_time - start_time) > 0 else 0
                progress = (i + 1) / total_iters * 100 # 计算训练进度百分比
                print(f"[Epoch{show_epoch:03d} {i+1:04d}/{total_iters}] 损失:{loss_value_scalar:.3f} | LR:{current_lr:.6f} | {speed:.1f}it/s ({progress:.1f}%)")
        
        # TensorBoard日志写入 - 只记录核心指标，减小日志体积
        # 【批次输出频率】TensorBoard日志写入规则：每10个批次写入一次，降低IO开销
        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Train/总损失', loss_value_scalar, global_step)
            writer.add_scalar('Train/学习率', optimizer.param_groups[0]["lr"], global_step)
            # 写入核心损失组件
            for k, v in filtered_loss_dict.items():
                writer.add_scalar(f'Train/{k}', v, global_step)
    
    # 训练结束，关闭进度条
    if pbar:
        pbar.close()
    
    # ========== 训练轮次结束 - 醒目尾部分隔+本轮总结 ==========
    avg_loss = total_loss / total_iters if total_iters > 0 else 0.0
    if dist_utils.is_main_process():
        # 计算当前epoch的总耗时
        epoch_time = time.time() - start_time
        print("" + "-"*100)
        current_sys_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{current_sys_time}✅✅第【{show_epoch:03d}】轮训练完成✅✅".center(80))
        print("-"*100)
        print(f"📊平均总损失: {avg_loss:.4f}")
        print(f"📊当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"📊本轮训练耗时: {epoch_time:.1f}s ({epoch_time/60:.1f}min)")
        print(f"📊平均速度: {total_iters/epoch_time:.2f} it/s")
        # ===================== 更新本轮最优训练损失指标 =====================
        global BEST_METRICS
        # 平均损失越小越好，判断是否更新最优值
        if avg_loss < BEST_METRICS["best_avg_loss"]:
            BEST_METRICS["best_avg_loss"] = avg_loss
            BEST_METRICS["best_avg_loss_epoch"] = show_epoch
            print(f"📊目前最优训练损失第{show_epoch:03d}轮: {avg_loss:.4f}")
        else:
            # ✅ 优化要求3：如果当前轮次损失不是最优，必打印历史最优值+轮次
            print(f"📊目前最优训练损失第{BEST_METRICS['best_avg_loss_epoch']:03d}轮: {BEST_METRICS['best_avg_loss']:.4f}")
        
        print("-"*100 + "")
    
    # 返回当前epoch的平均损失，供外部训练主函数调用
    return {'loss': avg_loss}


@torch.no_grad()  # 装饰器：关闭梯度计算，加速评估+减少显存占用，评估阶段无需梯度
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, 
             coco_evaluator: CocoEvaluator, device, epoch: int):  # ✅ 核心修复1：删除默认值=0，强制传参，彻底解决轮次显示000问题
    """
    模型评估函数
    Args:
        model: 待评估模型
        criterion: 损失函数
        postprocessor: 预测结果后处理器
        data_loader: 评估数据加载器
        coco_evaluator: COCO评估器
        device: 评估设备
        epoch: 当前训练轮次（必须传入，无默认值，用于记录最优指标对应轮次）
    Returns:
        dict: 评估指标
        CocoEvaluator: 更新后的评估器
    """
    # 模型切换为评估模式：关闭梯度计算、BN层固定、Dropout关闭等
    model.eval()
    # 损失函数切换为评估模式
    criterion.eval()
    # 清空评估器历史结果，避免累计上一轮评估数据
    coco_evaluator.cleanup()
    
    # 检测是否在nohup后台运行
    is_nohup = not sys.stdout.isatty()
    
    # 记录评估开始时间，用于计算评估耗时
    start_time = time.time()

    # ========== 核心修改点2：评估轮次显示 +1 ，从001开始 ==========
    show_epoch = epoch + 1
    # ========== ✅ 核心修复3：best_stat专用的epoch值，和显示轮次完全一致 ==========
    best_stat_epoch = show_epoch
    
    # ========== 评估阶段 醒目分隔+精简 ==========
    if dist_utils.is_main_process():
        print("" + "="*100)
        # ✅ 优化要求1：评估开始前 打印当前系统时间 - 新增核心代码
        current_sys_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{current_sys_time}📈📈第【{show_epoch:03d}】轮评估📈📈".center(80)) # ✅ 核心修复2：确保传入的epoch正常渲染
        print("="*100)
        print(f"📦 评估批次总数: {len(data_loader)}")
        print(f"📦 批次大小: {data_loader.batch_size}")
        print("-"*100 + "")
        
    
    # 初始化评估进度条，前台模式显示，后台模式关闭
    eval_pbar = None
    if dist_utils.is_main_process() and not is_nohup:
        eval_pbar = tqdm(total=len(data_loader), 
                        desc=f"评估进度 Epoch{show_epoch:03d}", # ✅ 优化：进度条显示当前评估轮次
                        ncols=100,
                        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                        colour='cyan')
    
    # 初始化评估日志器，仅做分布式同步用，不打印冗余日志
    metric_logger = MetricLogger(delimiter="  ")
    last_print_time = start_time

    # 遍历评估数据集，逐批次评估
    for i, (samples, targets) in enumerate(data_loader):
        # 数据转移到指定设备，non_blocking加速传输
        samples = samples.to(device, non_blocking=True)
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        # 模型前向推理，评估阶段不传入标注
        outputs = model(samples)
        # 获取原始图像尺寸，用于将预测框从缩放尺寸还原到原始尺寸
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # 预测结果后处理：坐标还原、置信度过滤、NMS非极大值抑制等
        results = postprocessor(outputs, orig_target_sizes)
        
        # 构建评估结果字典，key=图像ID，value=预测结果
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res) # 更新评估器结果
        
        # 评估进度输出 - 极简无冗余
        current_time = time.time()
        if eval_pbar:
            # 【批次输出频率】评估进度条更新规则：
            # 1. 距离上次更新超1秒  2. 每20个批次强制更新  3. 最后一个批次强制更新
            if current_time - last_print_time > 1 or i % 20 == 0 or i == len(data_loader) - 1:
                speed = (i + 1) / (current_time - start_time) if (current_time - start_time) > 0 else 0
                eval_pbar.set_postfix_str(f"{speed:.1f}it/s")
                last_print_time = current_time
            eval_pbar.update(1)
        
        elif is_nohup and dist_utils.is_main_process():
            # 【批次输出频率】评估后台日志打印规则：
            # 1. 每300个批次打印一次  2. 最后一个批次强制打印
            if i % 300 == 0 or i == len(data_loader) - 1:
                progress = (i + 1) / len(data_loader) * 100
                speed = (i + 1) / (current_time - start_time) if (current_time - start_time) > 0 else 0
                print(f"[评估进度 Epoch{show_epoch:03d} {i+1:04d}/{len(data_loader)}] {progress:.1f}% | {speed:.1f}it/s") # ✅ 优化：日志显示评估轮次

    
    # 评估结束，关闭进度条
    if eval_pbar:
        eval_pbar.close()
        
    print("-"*100 + "")
    
    # 计算评估总耗时
    eval_time = time.time() - start_time
    if dist_utils.is_main_process():
        print(f"📦 评估完成，评估耗时: {eval_time:.1f}s ({eval_time/60:.1f}min)")
    
    # 分布式同步：聚合多卡的评估结果，保证评估指标一致
    metric_logger.synchronize_between_processes()
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()   # 累计所有评估结果
        coco_evaluator.summarize()    # 计算最终评估指标

    # 初始化评估指标字典
    stats = {}
    # ========== 只保留COCO BBOX核心评估指标，删除所有无用指标 ==========
    if coco_evaluator is not None and 'bbox' in coco_evaluator.iou_types:
        bbox_stats = coco_evaluator.coco_eval['bbox'].stats.tolist()
        stats['coco_eval_bbox'] = bbox_stats
        
        # ========== ✅ 核心修改：已完全删除 ↓ 这行 best_stat 打印内容 ↓ ==========
        # print(f"\nbest_stat: {best_stat}")
        
        # 评估结果输出：醒目+精简+只显示核心指标
        if dist_utils.is_main_process():
            print("="*150)
            current_sys_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{current_sys_time}🏆🏆第【{show_epoch:03d}】轮COCO核心评估结果🏆🏆".center(80))
            print("="*127 + "🏆🏆" + f"第【{show_epoch:03d}】轮结果" + "🏆🏆")
            print(f"🎯        mAP@0.5 : {bbox_stats[1]:.4f}")
            print(f"🎯       mAP@0.75 : {bbox_stats[2]:.4f}")
            print(f"🎯 mAP@[0.5:0.95] : {bbox_stats[0]:.4f}")
            print("="*127 + "🏆🏆" + f"第【{show_epoch:03d}】轮结果" + "🏆🏆")
            
            # ===================== 更新全局最优评估指标 + ✅ 优化要求3核心逻辑 =====================
            global BEST_METRICS
            ap_05 = bbox_stats[1]   # AP@0.5
            ap_75 = bbox_stats[2]   # AP@0.75
            ap_all = bbox_stats[0]  # AP@[0.5:0.95]
            
            # 1. 更新最优值（原有逻辑保留）
            if ap_05 > BEST_METRICS["best_ap_05"]:
                BEST_METRICS["best_ap_05"] = ap_05
                BEST_METRICS["best_ap_05_epoch"] = show_epoch
            if ap_75 > BEST_METRICS["best_ap_75"]:
                BEST_METRICS["best_ap_75"] = ap_75
                BEST_METRICS["best_ap_75_epoch"] = show_epoch
            if ap_all > BEST_METRICS["best_ap_all"]:
                BEST_METRICS["best_ap_all"] = ap_all
                BEST_METRICS["best_ap_all_epoch"] = show_epoch

            # ✅ 优化要求3：必打印所有最优指标，无论是否刷新，无任何隐藏！
            print(f"🔥        最优mAP@0.5 : {BEST_METRICS['best_ap_05']:.4f}   | 记录于第 {BEST_METRICS['best_ap_05_epoch']:03d} 轮")
            print(f"🔥       最优mAP@0.75 : {BEST_METRICS['best_ap_75']:.4f}   | 记录于第 {BEST_METRICS['best_ap_75_epoch']:03d} 轮")
            print(f"🔥 最优mAP@[0.5:0.95] : {BEST_METRICS['best_ap_all']:.4f}   | 记录于第 {BEST_METRICS['best_ap_all_epoch']:03d} 轮")

            print("="*150)

    return stats, coco_evaluator

# ===================== 训练完成后调用 - 全局最优指标总结 =====================
def print_train_final_summary():
    """训练完全结束后，调用此函数即可打印全局最优指标总结报告"""
    if dist_utils.is_main_process():
        print("\n" + "🎉"*150)
        print("🏆🏆🏆🏆🏆 【训练全局最优指标总结报告】 🏆🏆🏆🏆🏆".center(80))
        print("🎉"*150)
        print("="*100)
        print(f"📉 最优训练平均总损失: {BEST_METRICS['best_avg_loss']:.4f}")
        print(f"🔍 最优损失对应轮次: 第 {BEST_METRICS['best_avg_loss_epoch']:03d} 轮")
        print("-"*100)
        print(f"📈 最优主指标mAP@0.5: {BEST_METRICS['best_ap_05']:.4f}")
        print(f"🔍 最优mAP@0.5对应轮次: 第 {BEST_METRICS['best_ap_05_epoch']:03d} 轮")
        print("-"*100)
        print(f"📈 最优高精度mAP@0.75: {BEST_METRICS['best_ap_75']:.4f}")
        print(f"🔍 最优mAP@0.75对应轮次: 第 {BEST_METRICS['best_ap_75_epoch']:03d} 轮")
        print("="*100)
        print(f"📈 最优综合mAP@[0.5:0.95]: {BEST_METRICS['best_ap_all']:.4f}")
        print(f"🔍 最优综合mAP对应轮次: 第 {BEST_METRICS['best_ap_all_epoch']:03d} 轮")
        print("-"*100)
        print("🎊🎊🎊 训练流程全部完成，恭喜！🎊🎊🎊\n")
        