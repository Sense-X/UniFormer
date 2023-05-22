# modified from https://github.com/facebookresearch/deit
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        outputs = model(samples)
        loss_avg = criterion(samples, outputs[0], targets)
        loss_cls = criterion(samples, outputs[1], targets)
        loss = 0.5 * loss_avg + 0.5 * loss_cls

        loss_avg_value = loss_avg.item()
        loss_cls_value = loss_cls.item()
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss_avg=loss_avg_value)
        metric_logger.update(loss_cls=loss_cls_value)
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def load_ema(model_ema, model):
    ema_state_dict = model_ema.ema.state_dict()
    for k, v in model.state_dict().items():
        k = k.replace('module.', '')
        if not k in ema_state_dict:
            if 'total_params' in k or  'total_ops' in k:
                continue
            raise ValueError(f'{k} not in ema_state_dict')
        tmp = v.data.clone()
        v.data.copy_(ema_state_dict[k].data)
        ema_state_dict[k].data.copy_(tmp)


def recover(model_ema, model):
    state_dict = model.state_dict()
    ema_state_dict = model_ema.ema.state_dict()
    for k, v in ema_state_dict.items():
        k_module = 'module.' + k
        if not k in state_dict and k_module in state_dict:
            k = k_module
        if not k in state_dict or not k_module in state_dict:
            raise ValueError(f'{k} not in state_dict')
        tmp = v.data.clone()
        v.data.copy_(state_dict[k].data)
        state_dict[k].data.copy_(tmp)


@torch.no_grad()
def evaluate(data_loader, model, device, model_ema=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # model.pos_embed.fill_(0)
    # switch to evaluation mode
    model.eval()

    if model_ema is not None:
        load_ema(model_ema, model)

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    if model_ema is not None:
        recover(model_ema, model)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
