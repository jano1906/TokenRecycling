import argparse
import platform
import random
import sys
import time

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
#from pytorch_lightning.plugins import NativeMixedPrecisionPlugin
from pytorch_lightning.strategies import DDPStrategy
from torch.cuda.amp import GradScaler

from utils.prepare import experiment_from_args
import numpy as np
import os


random.seed(1)
torch.manual_seed(1)
np.random.seed(1)


def define_args(parent_parser):
    parser = parent_parser.add_argument_group('train.py')
    parser.add_argument('--use-fp16',
                        help='sets models precision to FP16. Default is FP32',
                        action='store_true',
                        default=False)
    parser.add_argument('--load-model-path',
                        help='load model from pth',
                        type=str,
                        default=None)
    parser.add_argument('--wandb',
                        help='log to wandb',
                        type=bool,
                        default=False,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--tensorboard',
                        help='log to tensorboard',
                        type=bool,
                        default=False,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--ddp',
                        help='use DDP acceleration strategy',
                        type=bool,
                        default=False,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--name',
                        help='experiment name',
                        type=str,
                        default=None)
    parser.add_argument('--output-dir',
                        help='ouput directory for logs and checkpoints',
                        type=str,
                        default="")
    return parent_parser


def main():
    data_module, model, args = experiment_from_args(sys.argv, add_argparse_args_fn=define_args)

    plugins = []
    if args.use_fp16:
        grad_scaler = GradScaler()
        plugins += [NativeMixedPrecisionPlugin(precision=16, device='cuda', scaler=grad_scaler)]

    run_name = args.name
    if run_name is None:
        run_name = f'{time.strftime("%Y-%m-%d_%H:%M:%S")}-{platform.node()}'
    print('Run name:', run_name)

    loggers = []
    if args.tensorboard:
        loggers.append(TensorBoardLogger(save_dir=os.path.join(args.output_dir, "logs"), name=run_name))
    if args.wandb:
        loggers.append(WandbLogger(project='glimpse_mae', entity="ideas_cv", name=run_name))

    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(args.output_dir, "checkpoints", run_name), monitor="val/loss")
    early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=10, verbose=True, mode="min")

    trainer = Trainer(plugins=plugins, max_epochs=args.epochs, accelerator='auto', logger=loggers,
                      callbacks=[checkpoint_callback, early_stop_callback, RichProgressBar(leave=True), RichModelSummary(max_depth=3)],
                      enable_model_summary=False,
                      strategy=DDPStrategy(find_unused_parameters=False) if args.ddp else "auto")

    trainer.fit(model=model, datamodule=data_module, ckpt_path=args.load_model_path)

    if data_module.has_test_data:
        trainer.test(ckpt_path='best', datamodule=data_module)


if __name__ == "__main__":
    main()
