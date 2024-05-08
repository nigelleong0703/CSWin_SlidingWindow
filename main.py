# from CSwin import CSwin
import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torchvision.datasets import CocoDetection
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from timm.data import create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import load_checkpoint, create_model, resume_checkpoint
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import time
import os
import logging
from tqdm import tqdm

# import models
# import CSWin
from CSwin import CSwin
from model_config import *
import torch.nn as nn
from timm.utils import *

# from checkpoint_saver import CheckpointSaver
from timm.utils.checkpoint_saver import CheckpointSaver

config_parser = parser = argparse.ArgumentParser()

# Dataset / Model parameters
parser.add_argument(
    "--data",
    default="./dataset",
    metavar="DIR",
    help="path to dataset",
)
# parser.add_argument(
#     "--model",
#     default="CSWin_64_12211_tiny_224_norm",
#     type=str,
#     metavar="MODEL",
#     help='Name of model to train (default: "countception"',
# )
parser.add_argument(
    "--model",
    default="norm",
    type=str,
    metavar="MODEL",
    help='Name of model to train ("norm", "sw1", "sw2")',
)
parser.add_argument(
    "--pretrained",
    action="store_true",
    default=False,
    help="Start with pretrained version of specified network (if avail)",
)
parser.add_argument(
    "--initial-checkpoint",
    default="",
    type=str,
    metavar="PATH",
    help="Initialize model from this checkpoint (default: none)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="Resume full model and optimizer state from checkpoint (default: none)",
)
parser.add_argument(
    "--eval_checkpoint",
    default="",
    type=str,
    metavar="PATH",
    help="path to eval checkpoint (default: none)",
)
parser.add_argument(
    "--no-resume-opt",
    action="store_true",
    default=False,
    help="prevent resume of optimizer state when resuming model",
)
parser.add_argument(
    "--num-classes",
    type=int,
    default=100,
    metavar="N",
    help="number of label classes (default: 100)",
)
parser.add_argument(
    "--gp",
    default=None,
    type=str,
    metavar="POOL",
    help="Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.",
)
parser.add_argument(
    "--img-size",
    type=int,
    default=224,
    metavar="N",
    help="Image patch size (default: None => model default)",
)
parser.add_argument(
    "--crop-pct",
    default=None,
    type=float,
    metavar="N",
    help="Input image center crop percent (for validation only)",
)
parser.add_argument(
    "--mean",
    type=float,
    nargs="+",
    default=None,
    metavar="MEAN",
    help="Override mean pixel value of dataset",
)
parser.add_argument(
    "--std",
    type=float,
    nargs="+",
    default=None,
    metavar="STD",
    help="Override std deviation of of dataset",
)
parser.add_argument(
    "--interpolation",
    default="",
    type=str,
    metavar="NAME",
    help="Image resize interpolation type (overrides model)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size for training (default: 128)",
)
parser.add_argument(
    "-vb",
    "--validation-batch-size-multiplier",
    type=int,
    default=1,
    metavar="N",
    help="ratio of validation batch size to training batch size (default: 1)",
)

# Optimizer parameters
parser.add_argument(
    "--opt",
    default="adamw",
    type=str,
    metavar="OPTIMIZER",
    help='Optimizer (default: "adamw"',
)
parser.add_argument(
    "--opt-eps",
    default=None,
    type=float,
    metavar="EPSILON",
    help="Optimizer Epsilon (default: None, use opt default)",
)
parser.add_argument(
    "--opt-betas",
    default=None,
    type=float,
    nargs="+",
    metavar="BETA",
    help="Optimizer Betas (default: None, use opt default)",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="Optimizer momentum (default: 0.9)",
)
parser.add_argument(
    "--weight-decay",
    type=float,
    default=0.05,
    help="weight decay (default: 0.005 for adamw)",
)
parser.add_argument(
    "--clip-grad",
    type=float,
    default=None,
    metavar="NORM",
    help="Clip gradient norm (default: None, no clipping)",
)

# Learning rate schedule parameters
parser.add_argument(
    "--sched",
    default="cosine",
    type=str,
    metavar="SCHEDULER",
    help='LR scheduler (default: "cosine"',
)
parser.add_argument(
    "--lr", type=float, default=5e-4, metavar="LR", help="learning rate (default: 0.01)"
)
parser.add_argument(
    "--lr-noise",
    type=float,
    nargs="+",
    default=None,
    metavar="pct, pct",
    help="learning rate noise on/off epoch percentages",
)
parser.add_argument(
    "--lr-noise-pct",
    type=float,
    default=0.67,
    metavar="PERCENT",
    help="learning rate noise limit percent (default: 0.67)",
)
parser.add_argument(
    "--lr-noise-std",
    type=float,
    default=1.0,
    metavar="STDDEV",
    help="learning rate noise std-dev (default: 1.0)",
)
parser.add_argument(
    "--lr-cycle-mul",
    type=float,
    default=1.0,
    metavar="MULT",
    help="learning rate cycle len multiplier (default: 1.0)",
)
parser.add_argument(
    "--lr-cycle-limit",
    type=int,
    default=1,
    metavar="N",
    help="learning rate cycle limit",
)
parser.add_argument(
    "--warmup-lr",
    type=float,
    default=1e-6,
    metavar="LR",
    help="warmup learning rate (default: 0.0001)",
)
parser.add_argument(
    "--min-lr",
    type=float,
    default=1e-5,
    metavar="LR",
    help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=300,
    metavar="N",
    help="number of epochs to train (default: 2)",
)
parser.add_argument(
    "--start-epoch",
    default=None,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "--decay-epochs",
    type=float,
    default=30,
    metavar="N",
    help="epoch interval to decay LR",
)
parser.add_argument(
    "--warmup-epochs",
    type=int,
    default=20,
    metavar="N",
    help="epochs to warmup LR, if scheduler supports",
)
parser.add_argument(
    "--cooldown-epochs",
    type=int,
    default=10,
    metavar="N",
    help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
)
parser.add_argument(
    "--patience-epochs",
    type=int,
    default=10,
    metavar="N",
    help="patience epochs for Plateau LR scheduler (default: 10",
)
parser.add_argument(
    "--decay-rate",
    "--dr",
    type=float,
    default=0.1,
    metavar="RATE",
    help="LR decay rate (default: 0.1)",
)

# Augmentation & regularization parameters
parser.add_argument(
    "--no-aug",
    action="store_true",
    default=False,
    help="Disable all training augmentation, override other train aug args",
)
parser.add_argument(
    "--scale",
    type=float,
    nargs="+",
    default=[0.08, 1.0],
    metavar="PCT",
    help="Random resize scale (default: 0.08 1.0)",
)
parser.add_argument(
    "--ratio",
    type=float,
    nargs="+",
    default=[3.0 / 4.0, 4.0 / 3.0],
    metavar="RATIO",
    help="Random resize aspect ratio (default: 0.75 1.33)",
)
parser.add_argument(
    "--hflip", type=float, default=0.5, help="Horizontal flip training aug probability"
)
parser.add_argument(
    "--vflip", type=float, default=0.0, help="Vertical flip training aug probability"
)
parser.add_argument(
    "--color-jitter",
    type=float,
    default=0.4,
    metavar="PCT",
    help="Color jitter factor (default: 0.4)",
)
parser.add_argument(
    "--aa",
    type=str,
    default="rand-m9-mstd0.5-inc1",
    metavar="NAME",
    help='Use AutoAugment policy. "v0" or "original". (default: None)',
),
parser.add_argument(
    "--aug-splits",
    type=int,
    default=0,
    help="Number of augmentation splits (default: 0, valid: 0 or >=2)",
)
parser.add_argument(
    "--jsd",
    action="store_true",
    default=False,
    help="Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.",
)
parser.add_argument(
    "--reprob",
    type=float,
    default=0.25,
    metavar="PCT",
    help="Random erase prob (default: 0.25)",
)
parser.add_argument(
    "--remode", type=str, default="pixel", help='Random erase mode (default: "const")'
)
parser.add_argument(
    "--recount", type=int, default=1, help="Random erase count (default: 1)"
)
parser.add_argument(
    "--resplit",
    action="store_true",
    default=False,
    help="Do not random erase first (clean) augmentation split",
)
parser.add_argument(
    "--mixup",
    type=float,
    default=0.8,
    help="mixup alpha, mixup enabled if > 0. (default: 0.)",
)
parser.add_argument(
    "--cutmix",
    type=float,
    default=1.0,
    help="cutmix alpha, cutmix enabled if > 0. (default: 0.)",
)
parser.add_argument(
    "--cutmix-minmax",
    type=float,
    nargs="+",
    default=None,
    help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
)
parser.add_argument(
    "--mixup-prob",
    type=float,
    default=1.0,
    help="Probability of performing mixup or cutmix when either/both is enabled",
)
parser.add_argument(
    "--mixup-switch-prob",
    type=float,
    default=0.5,
    help="Probability of switching to cutmix when both mixup and cutmix enabled",
)
parser.add_argument(
    "--mixup-mode",
    type=str,
    default="batch",
    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
)
parser.add_argument(
    "--mixup-off-epoch",
    default=0,
    type=int,
    metavar="N",
    help="Turn off mixup after this epoch, disabled if 0 (default: 0)",
)
parser.add_argument(
    "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
)
parser.add_argument(
    "--train-interpolation",
    type=str,
    default="random",
    help='Training interpolation (random, bilinear, bicubic default: "random")',
)
parser.add_argument(
    "--drop", type=float, default=0.0, metavar="PCT", help="Dropout rate (default: 0.0)"
)
parser.add_argument(
    "--drop-connect",
    type=float,
    default=None,
    metavar="PCT",
    help="Drop connect rate, DEPRECATED, use drop-path (default: None)",
)
parser.add_argument(
    "--drop-path",
    type=float,
    default=0.1,
    metavar="PCT",
    help="Drop path rate (default: None)",
)
parser.add_argument(
    "--drop-block",
    type=float,
    default=None,
    metavar="PCT",
    help="Drop block rate (default: None)",
)

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument(
    "--bn-tf",
    action="store_true",
    default=False,
    help="Use Tensorflow BatchNorm defaults for models that support it (default: False)",
)
parser.add_argument(
    "--bn-momentum",
    type=float,
    default=None,
    help="BatchNorm momentum override (if not None)",
)
parser.add_argument(
    "--bn-eps",
    type=float,
    default=None,
    help="BatchNorm epsilon override (if not None)",
)
parser.add_argument(
    "--sync-bn",
    action="store_true",
    help="Enable NVIDIA Apex or Torch synchronized BatchNorm.",
)
parser.add_argument(
    "--dist-bn",
    type=str,
    default="",
    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")',
)
parser.add_argument(
    "--split-bn",
    action="store_true",
    help="Enable separate BN layers per augmentation split.",
)

# Model Exponential Moving Average
parser.add_argument(
    "--model-ema",
    action="store_true",
    default=True,
    help="Enable tracking moving average of model weights",
)
parser.add_argument(
    "--model-ema-force-cpu",
    action="store_true",
    default=False,
    help="Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.",
)
parser.add_argument(
    "--model-ema-decay",
    type=float,
    default=0.99992,
    help="decay factor for model weights moving average (default: 0.9998)",
)

# Misc
parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=50,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--recovery-interval",
    type=int,
    default=0,
    metavar="N",
    help="how many batches to wait before writing recovery checkpoint",
)
parser.add_argument(
    "-j",
    "--workers",
    type=int,
    default=8,
    metavar="N",
    help="how many training processes to use (default: 1)",
)
parser.add_argument("--num-gpu", type=int, default=1, help="Number of GPUS to use")
parser.add_argument(
    "--save-images",
    action="store_true",
    default=False,
    help="save images of input bathes every log interval for debugging",
)
parser.add_argument(
    "--amp",
    action="store_true",
    default=False,
    help="use NVIDIA Apex AMP or Native AMP for mixed precision training",
)
# parser.add_argument(
#     "--apex-amp",
#     action="store_true",
#     default=FaBatchlse,
#     help="Use NVIDIA Apex AMP mixed precision",
# )
parser.add_argument(
    "--native-amp",
    action="store_true",
    default=False,
    help="Use Native Torch AMP mixed precision",
)
parser.add_argument(
    "--channels-last",
    action="store_true",
    default=False,
    help="Use channels_last memory layout",
)
parser.add_argument(
    "--pin-mem",
    action="store_true",
    default=False,
    help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
)
parser.add_argument(
    "--no-prefetcher",
    action="store_true",
    default=False,
    help="disable fast prefetcher",
)
parser.add_argument(
    "--output",
    default="",
    type=str,
    metavar="PATH",
    help="path to output folder (default: none, current dir)",
)
parser.add_argument(
    "--eval-metric",
    default="top1",
    type=str,
    metavar="EVAL_METRIC",
    help='Best metric (default: "top1"',
)
parser.add_argument(
    "--tta",
    type=int,
    default=0,
    metavar="N",
    help="Test/inference time augmentation (oversampling) factor. 0=None (default: 0)",
)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument(
    "--use-multi-epochs-loader",
    action="store_true",
    default=False,
    help="use the multi-epochs-loader to save time at the beginning of every epoch",
)

parser.add_argument(
    "--use-chk",
    action="store_true",
    default=False,
    help="Enable tracking moving average of model weights",
)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _parse_args():
    # Do we have a config file to parse?
    # args_config, remaining = config_parser.parse_known_args()
    # if args_config.config:
    #     with open(args_config.config, "r") as f:
    #         cfg = yaml.safe_load(f)
    #         parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args()

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    args, args_text = _parse_args()
    args.device = "cuda:0"
    # torch.cuda.set_device(args.device)

    # torch.manual_seed(args.seed + args.rank)
    torch.manual_seed(args.seed)
    
    # model = CSWin_64_12211_tiny_224().cuda()
    if args.model == "norm":
        model = CSWin_64_12211_tiny_224_norm().cuda()
    elif args.model == "sw1":
        model = CSWin_64_12211_tiny_224_sw1().cuda()
    elif args.model == "sw2":
        model = CSWin_64_12211_tiny_224_sw2().cuda()
    else:
        print("Invalid model")
        return
    print(model.default_cfg)
    model.default_cfg["num_classes"] = 100
    print(model.default_cfg)
    print("Num:", count_parameters(model) / 1e6)
    print(model)

    optimizer = create_optimizer(args, model)
    loss_scaler = None

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)

    train_dir = os.path.join(args.data, "train")
    test_dir = os.path.join(args.data, "test")
    eval_dir = os.path.join(args.data, "val")

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    train_mean = [0.4829, 0.4550, 0.4016]
    train_std = [0.2726, 0.2667, 0.2782]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1829, 0.4550, 0.4016), (0.2726, 0.2667, 0.2782)),
    ])
    train_dataset = ImageFolder(root=train_dir, transform=transform)
    # loader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None

    train_interpolation = args.train_interpolation
    # if args.no_aug or not train_interpolation:
    #     train_interpolation = data_config['interpolation']
    loader_train = create_loader(
        train_dataset,
        input_size=(3, 224, 224),
        batch_size=args.batch_size,
        is_training=True,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        # interpolation=train_interpolation,
        mean=(0.1829, 0.4550, 0.4016),
        std=(0.2726, 0.2667, 0.2782),
        collate_fn=collate_fn,
    )

    eval_dataset = ImageFolder(root=eval_dir, transform=transform)
    # loader_eval = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    loader_eval = create_loader(
        eval_dataset,
        input_size=(3, 224, 224),
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        is_training=False,
        # interpolation=data_config['interpolation'],
        mean=(0.1829, 0.4550, 0.4016),
        std=(0.2726, 0.2667, 0.2782),
        # crop_pct=data_config['crop_pct'],
    )

    test_dataset = ImageFolder(root=test_dir, transform=transform)
    loader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    train_loss_fn = nn.CrossEntropyLoss().cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # dataset_eval = McDataset(args.data, './dataset/ILSVRC2012_name_val.txt', 'val')

    # loader_eval = create_loader(
    #     dataset_eval,
    #     input_size=data_config['input_size'],
    #     batch_size=args.validation_batch_size_multiplier * args.batch_size,
    #     is_training=False,
    #     use_prefetcher=args.prefetcher,
    #     interpolation=data_config['interpolation'],
    #     mean=data_config['mean'],
    #     std=data_config['std'],
    #     num_workers=args.workers,
    #     distributed=args.distributed,
    #     crop_pct=data_config['crop_pct'],
    #     pin_memory=args.pin_mem,
    # )

    # if args.eval_checkpoint:  # evaluate the model
    #     load_checkpoint(model, args.eval_checkpoint, args.model_ema)
    #     val_metrics = validate(model, loader_eval, validate_loss_fn, args)
    #     print(f"Top-1 accuracy of the model is: {val_metrics['top1']:.1f}%")
    #     return

    

    amp_autocast = suppress
    loss_scaler = None
    model_ema = None
    mixup_fn = None

    saver = None
    output_dir = ""
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    decreasing = True if eval_metric == "loss" else False

    if mixup_active:
        mixup_args = dict(
        mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
        prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
        label_smoothing=args.smoothing, num_classes=args.num_classes)
        mixup_fn = Mixup(**mixup_args)

    saver = None
    saver = CheckpointSaver(
        model=model,
        optimizer=optimizer,
        args=args,
        model_ema=model_ema,
        amp_scaler=loss_scaler,
        checkpoint_dir=output_dir,
        recovery_dir=output_dir,
        decreasing=decreasing,
    )
    with open(os.path.join(output_dir, "args.yaml"), "w") as f:
        f.write(args_text)

    if mixup_active:
    # smoothing is handled with mixup target transform
        train_loss_fn = SoftTargetCrossEntropy().cuda()
    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
    else:
        train_loss_fn = nn.CrossEntropyLoss().cuda()
        validate_loss_fn = nn.CrossEntropyLoss().cuda()

    for epoch in range(start_epoch, num_epochs):
        train_metrics = train_epoch(
            epoch,
            model,
            loader_train,
            optimizer,
            train_loss_fn,
            args,
            lr_scheduler=lr_scheduler,
            saver=saver,
            output_dir=output_dir,
            amp_autocast=amp_autocast,
            loss_scaler=loss_scaler,
            model_ema=model_ema,
            mixup_fn=mixup_fn,
        )
        print(train_metrics)

        eval_metrics = validate(
            model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast
        )

        if lr_scheduler is not None:
            # step LR for next epoch
            lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

        update_summary(
            epoch,
            train_metrics,
            eval_metrics,
            os.path.join(output_dir, "summary.csv"),
            write_header=best_metric is None,
        )

        print(
            "Epoch: {}, Train Metrics: {}, Eval Metrics: {}".format(
                epoch,
                train_metrics,
                eval_metrics,
            )
        )

        if saver is not None:
            # save proper checkpoint with eval metric
            save_metric = eval_metrics[eval_metric]
            best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)


def train_epoch(
    epoch,
    model,
    loader,
    optimizer,
    loss_fn,
    args,
    lr_scheduler=None,
    saver=None,
    output_dir="",
    amp_autocast=suppress,
    loss_scaler=None,
    model_ema=None,
    mixup_fn=None,
):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False
        
    second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    progress_bar = tqdm(enumerate(loader), total=len(loader))
    for batch_idx, (input, target) in progress_bar:
        input = input.cuda()
        target = target.cuda()
        if mixup_fn is not None:
            input, target = mixup_fn(input, target)
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)

        # Forward pass
        output = model(input)
        loss = loss_fn(output, target)
        losses_m.update(loss.item(), input.size(0))

        # Backward pass and optimization
        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                clip_grad=args.clip_grad,
                parameters=model.parameters(),
                create_graph=second_order,
            )
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        # Model EMA update
        if model_ema is not None:
            model_ema.update(model)

        num_updates += 1
        batch_time_m.update(time.time() - end)

        # Logging
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group["lr"] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.save_images and output_dir:
                torchvision.utils.save_image(
                    input,
                    os.path.join(output_dir, "train-batch-%d.jpg" % batch_idx),
                    padding=0,
                    normalize=True,
                )

        # Save recovery checkpoints
        if (
            saver is not None
            and args.recovery_interval
            and (last_batch or (batch_idx + 1) % args.recovery_interval == 0)
        ):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        # Learning rate scheduler step
        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()

    # Sync lookahead optimizer if exists
    if hasattr(optimizer, "sync_lookahead"):
        optimizer.sync_lookahead()

    return OrderedDict([("loss", losses_m.avg)])


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=""):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx

            input = input.cuda()
            target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            reduced_loss = loss.data

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()


    metrics = OrderedDict(
        [("loss", losses_m.avg), ("top1", top1_m.avg), ("top5", top5_m.avg)]
    )

    return metrics


if __name__ == "__main__":
    main()
