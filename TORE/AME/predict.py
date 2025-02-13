import argparse
import os
import sys

import torch
import tqdm
import json
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar, RichModelSummary
from pytorch_lightning.strategies import DDPStrategy
from fvcore.nn import FlopCountAnalysis

from architectures.reconstruction import ReconstructionMae
from architectures.segmentation import SegmentationMae
from utils.prepare import experiment_from_args
import yaml
import numpy as np
import random

random.seed(1)
torch.manual_seed(1)
np.random.seed(1)

from utils.visualize import save_reconstructions, upscale_patch_values


def define_args(parent_parser):
    parser = parent_parser.add_argument_group('predict.py')
    parser.add_argument('--load-model-path',
                        help='load model from pth',
                        type=str,
                        required=True)
    parser.add_argument('--max-batches',
                        help='number of batches from dataset to process',
                        type=int,
                        default=99999999999999)
    parser.add_argument('--visualization-path',
                        help='path to save visualizations to',
                        type=str)
    parser.add_argument('--ddp',
                        help='use DDP acceleration strategy',
                        type=bool,
                        default=False,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--test',
                        help='do test',
                        type=bool,
                        default=False,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--split',
                        help='split to use',
                        choices=['train', 'val', 'test'],
                        type=str,
                        default=None)
    parser.add_argument('--dump-path',
                        help='do latent dump to file',
                        type=str,
                        default=None)
    parser.add_argument('--avg-glimpse-path',
                        help='do avg glimpse to file',
                        type=str,
                        default=None)
    parser.add_argument('--output_dir',
                        help='path to output directory',
                        type=str,
                        default="")
    parser.add_argument('--last-only',
                        help='save visualizations only for the last step',
                        type=bool,
                        default=False,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--eval-run',
                        help='path to log of a run to eval',
                        type=str,
                        default=None)
    parser.add_argument('--eval-ckpt',
                        help='numer of checkpoint in run directory to eval',
                        type=str,
                        default=None)
    parser.add_argument('--count-flops-path',
                        type=str,
                        default=None)
    parser.add_argument('--count-flops-mode',
                        type=str,
                        default="selection_cls")
    parser.add_argument('--sequential-accuracy',
                        type=bool,
                        default=False,
                        action=argparse.BooleanOptionalAction)
    return parent_parser


def do_visualizations(args, model, loader):
    print(model.load_state_dict(torch.load(args.load_model_path, map_location='cpu')['state_dict']))

    model = model.cuda()
    model.eval()
    model.debug = True

    #assert isinstance(model, ReconstructionMae) or isinstance(model, SegmentationMae)

    for idx, batch in enumerate(tqdm.tqdm(loader, total=min(args.max_batches, len(loader)))):
        out = model.predict_step([x.cuda() for x in batch], idx)
        clss = torch.stack([x["classification"] for x in out["steps"]])
        torch.save(clss, os.path.join("/home/jano1906/git/TORE/AME/viz/expl_viz", "classifications.pth"))
        torch.save(batch[1], os.path.join("/home/jano1906/git/TORE/AME/viz/expl_viz", "gt.pth"))
        save_reconstructions(model, out, batch, vis_id=idx, dump_path=args.visualization_path, last_only=args.last_only)
        return

    model.debug = False


def do_dump_latent(args, model, loader):
    os.makedirs(args.dump_path, exist_ok=True)
    model.load_pretrained_mae(args.load_model_path)

    model = model.cuda()
    model.eval()
    model.debug = True

    latents = []
    targets = []

    for idx, batch in enumerate(tqdm.tqdm(loader, total=min(args.max_batches, len(loader)))):
        if idx >= args.max_batches:
            break
        out = model.predict_step([x.cuda() for x in batch], idx)
        latents.append(out['steps'][-1]['latent'])
        targets.append(batch[1])
    latents = torch.cat(latents, dim=0)
    targets = torch.cat(targets, dim=0)
    torch.save({'latents': latents, 'targets': targets}, os.path.join(args.dump_path, f'embeds_{args.split}.pck'))

    model.debug = False


def do_avg_glimpse(args, model, loader):
    model.load_state_dict(torch.load(args.load_model_path, map_location='cpu')['state_dict'])

    model = model.cuda()
    model.eval()

    avg_mask = None
    items = 0

    for idx, batch in enumerate(tqdm.tqdm(loader)):
        if idx >= args.max_batches:
            break
        out = model.predict_step([x.cuda() for x in batch], idx)
        items += out['mask'].shape[0]
        if avg_mask is None:
            avg_mask = out['mask'].detach().clone().cpu().float().sum(dim=0)
        else:
            avg_mask += out['mask'].detach().clone().cpu().float().sum(dim=0)
    avg_mask /= items
    grid_h = model.mae.grid_size[0]
    grid_w = model.mae.grid_size[1]
    patch_size = model.mae.patch_embed.patch_size
    avg_mask = avg_mask.reshape(grid_h, grid_w).numpy()
    avg_mask = upscale_patch_values(avg_mask, patch_size)
    plt.imsave(args.avg_glimpse_path, avg_mask)


def do_test(args, model, loader):
    #assert args.arch in ["RandomClsMae", "AttentionClsMae"]
    trainer = Trainer(accelerator='auto', callbacks=[RichProgressBar(leave=True), RichModelSummary(max_depth=3)],
                      strategy=DDPStrategy(find_unused_parameters=False) if args.ddp else None)
    run_name = os.path.split(args.load_model_path)[0]
    logs_path = run_name.replace("checkpoints", "logs")
    with open(os.path.join(logs_path, "version_0", "hparams.yaml"), "r") as f:
        params = yaml.unsafe_load(f)
    
    ret = {}
    ret["training_K"] = params["args"].K
    ret["K"] = args.K
    ret["selector"] = "random" if args.arch == "RandomClsMae" else "attention"
    ret["train_selector"] = params["args"].arch
    ret["train_single_step"] = params["args"].single_step
    ret["arch"] = params["args"].vit_arch
    ret["train_sample_divisions"] = params["args"].sample_divisions
    ret["train_rec_loss"] = params["args"].rec_loss
    ret["train_glimpse_size"] = params["args"].glimpse_size
    ret["test_glimpse_size"] = args.glimpse_size
    ret["ckpt"] = args.eval_ckpt
    ret["force_K_prediction"] = args.force_K_prediction
    # -----------------------------------------------------
    #from architectures import RandomClsMae, AttentionClsMae
    #if not ret["train_sample_divisions"] and not ret["train_rec_loss"] and isinstance(model, RandomClsMae):
    #    if ret["K"]!=0:
    #        return
    #    pass
    #elif not ret["train_sample_divisions"] and ret["train_rec_loss"] and isinstance(model, AttentionClsMae):
    #    if ret["K"]!=0:
    #        return
    #    pass
    #elif ret["train_sample_divisions"] and ret["train_rec_loss"] and isinstance(model, AttentionClsMae):
    #    pass
    #else:
    #    return
    # -----------------------------------------------------

    score = trainer.test(model=model, ckpt_path=args.load_model_path, dataloaders=loader)
    for k in score[0]:
        if isinstance(k, str) and k.startswith("test/accuracy"):
            new_k = k.replace("test/accuracy", "acc")
            ret[new_k] = score[0][k]
        elif isinstance(k, str) and k.startswith("test/"):
            new_k = k.replace("test/", "")
            ret[new_k] = score[0][k]
    
    with open(os.path.join(args.output_dir, "log.json"), "a") as f:
        f.write(json.dumps(ret) + "\n")

def eval_run_args(args):
    if args.eval_run is not None:
        assert args.eval_ckpt is not None
        with open(os.path.join(args.eval_run, "version_0", "hparams.yaml"), "r") as f:
            params = yaml.unsafe_load(f)
        args.vit_arch = params["args"].vit_arch
        ckpt = args.eval_ckpt        
        args.load_model_path = ckpt
        args.pretrained_mae_path = ckpt
    return args

from torch import nn
from architectures.selectors import RandomGlimpseSelector
from architectures import AttentionClsMae

class MaeWrapper(nn.Module):
    def __init__(self, mae):
        super().__init__()
        self.mae = mae
    def forward(self, x, latent, mask, mask_ids, new_mask_ids, mode):
        if mode == "enc":
            out = self.mae.forward_encoder_next(x, new_mask_ids)
        elif mode == "dec":
            out = self.mae.forward_decoder(latent, mask, mask_ids)
        return out

class FLopsCountGlimpseMae(nn.Module):
    def __init__(self, model: AttentionClsMae, K, img_size, input=None, args=None):
        super().__init__()
        self.model = model
        self.attention_selector = model.glimpse_selector
        self.random_selector = RandomGlimpseSelector(model, args)
        self.mae = model.mae
        self.wrapped_mae = MaeWrapper(self.mae)
        self.selection = True
        self.classification = True
        self.device = "cpu"
        
        if input is not None:
            self.input = input
            self.mae._comp_next_cache = input["comp_next_cache"]
        else:
            self.mae.comp_next_init(K)
            self.input = input if input is not None else {
                "mask": torch.full((1, self.mae.grid_size[0] * self.mae.grid_size[1]), fill_value=False, device=self.device),
                "mask_ids": torch.empty((1, 0), dtype=torch.int32, device=self.device),
                "comp_next_cache": self.mae._comp_next_cache
            }
        self.img = torch.zeros(1,3,img_size[0],img_size[1])
        
        # zero step (initialize decoder attention weights)
        self.model.forward_one(self.img,
                                torch.empty((1, 0), dtype=torch.int32, device=self.device),
                                torch.full((1, self.mae.grid_size[0] * self.mae.grid_size[1]), fill_value=False, device=self.device),
                                [], [], [], initial_step=True)

    def set_mode_selection_cls(self):
        self.selection = True
        self.classification = True
    def set_mode_selection(self):
        self.selection = True
        self.classification = False
    def set_mode_cls(self):
        self.selection = False
        self.classification = True
    
    def forward(self):
        if self.selection:
            mask, mask_indices, glimpse, new_mask, new_mask_ids = self.attention_selector(self.input["mask"], self.input["mask_ids"], self.mae._comp_next_cache["i"])
        else:
            mask, mask_indices, glimpse, new_mask, new_mask_ids = self.random_selector(self.input["mask"], self.input["mask_ids"], self.mae._comp_next_cache["i"])
        self.input["mask"] = mask
        self.input["mask_ids"] = mask_indices

        x = self.model.cut_subimage(self.img, new_mask.squeeze(0))
        latent = self.wrapped_mae(x, None, mask, mask_indices, new_mask_ids, "enc")
        self.input["comp_next_cache"] = self.mae._comp_next_cache
        out, cls = torch.empty(0), torch.empty(0)
        if self.selection:
            out = self.wrapped_mae(x, latent, mask, mask_indices, new_mask_ids, "dec")

        if self.classification:
            cls = self.model.head(latent)
        return out, cls


def count_flops(args, datamodule, mode):
    def create_model_fn(K, input=None):
        model = AttentionClsMae(args, datamodule)
        flops_model = FLopsCountGlimpseMae(model, K, datamodule.image_size, input=input, args=args)
        if mode == "selection_cls":
            flops_model.set_mode_selection_cls()
        elif mode == "selection":
            flops_model.set_mode_selection()
        elif mode == "cls":
            flops_model.set_mode_cls()
        else:
            assert False
        
        return flops_model

    with torch.no_grad():
        flops = {}

        for k in tqdm.tqdm(range(13), f"K: "):
            input = None
            flops[k] = []
            for i in range(args.num_glimpses):
                model = create_model_fn(k, input=input)
                model.eval()
                flops[k].append(FlopCountAnalysis(model, ()).total())
                input = model.input
                cache = input["comp_next_cache"]
                assert len(cache["xs_feats"]) == i+1, len(cache["xs_feats"])
                assert cache["i"] == i+1, cache["i"]
            break
    return flops


def main():
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    data_module, model, args = experiment_from_args(sys.argv, add_argparse_args_fn=define_args, no_aug=True, modify_args_fn=eval_run_args)
    os.makedirs(args.output_dir, exist_ok=True)

    split = args.split
    if split is None:
        if data_module.has_test_data:
            split = 'test'
        else:
            split = 'val'
    if split == 'test':
        data_module.setup('test')
    else:
        data_module.setup('fit')
    args.split = split
    loader = {
        'train': data_module.train_dataloader,
        'val': data_module.val_dataloader,
        'test': data_module.test_dataloader
    }[split]()

    if args.visualization_path is not None:
        do_visualizations(args, model, loader)

    if args.test:
        do_test(args, model, loader)

    if args.dump_path is not None:
        do_dump_latent(args, model, loader)

    if args.avg_glimpse_path is not None:
        do_avg_glimpse(args, model, loader)
    
    if args.count_flops_path is not None:
        import pandas as pd
        flops = count_flops(args, data_module, args.count_flops_mode)
        flat_flops = [(str(k),i, v/1000000000) for k, l in flops.items() for i, v in enumerate(l)]
        df = pd.DataFrame(flat_flops, columns=["K", "i", "GFLOPs"])
        pd.DataFrame.to_csv(df, f"{args.count_flops_path}_{args.count_flops_mode}.csv")


if __name__ == "__main__":
    main()
