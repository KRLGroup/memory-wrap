# reference: https://github.com/omihub777/ViT-CIFAR/blob/main/main.py

import argparse
import torch
import torchvision
import pytorch_lightning as pl
import warmup_scheduler
import numpy as np
from utils import get_model, get_dataset, get_experiment_name, get_criterion
from da import CutMix, MixUp
from pytorch_lightning.trainer.supporters import CombinedLoader
import numpy as np
import random
import os
from typing import List

parser = argparse.ArgumentParser()
parser.add_argument("--api-key", help="API Key for Comet.ml")
parser.add_argument("--dataset", default="c10", type=str,
                    help="[c10, c100, svhn]")
parser.add_argument("--num-classes", default=10, type=int)
parser.add_argument("--model-name", default="encoder_vit",
                    help="[vit]", type=str)
parser.add_argument("--patch", default=8, type=int)
parser.add_argument("--batch-size", default=128, type=int)
parser.add_argument("--eval-batch-size", default=1024, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--min-lr", default=1e-5, type=float)
parser.add_argument("--beta1", default=0.9, type=float)
parser.add_argument("--beta2", default=0.999, type=float)
parser.add_argument("--off-benchmark", action="store_true")
parser.add_argument("--max-epochs", default=200, type=int)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--weight-decay", default=5e-5, type=float)
parser.add_argument("--warmup-epoch", default=5, type=int)
parser.add_argument("--precision", default=16, type=int)
parser.add_argument("--autoaugment", action="store_true")
parser.add_argument("--criterion", default="ce")
parser.add_argument("--label-smoothing", action="store_true")
parser.add_argument("--smoothing", default=0.1, type=float)
parser.add_argument("--rcpaste", action="store_true")
parser.add_argument("--cutmix", action="store_true")
parser.add_argument("--mixup", action="store_true")
parser.add_argument("--dropout", default=0.0, type=float)
parser.add_argument("--head", default=12, type=int)
parser.add_argument("--num-layers", default=7, type=int)
parser.add_argument("--hidden", default=384, type=int)
parser.add_argument("--mlp-hidden", default=384, type=int)
parser.add_argument("--off-cls-token", action="store_true")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--mem_samples", default=100, type=int)
parser.add_argument("--data_samples", default=1000, type=int)
parser.add_argument("--project-name", default="VisionTransformer")
parser.add_argument("--data_root", default="data", type=str)
args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
args.benchmark = True
args.gpus = torch.cuda.device_count()
args.num_workers = 4*args.gpus if args.gpus else 8
args.is_cls_token = True if not args.off_cls_token else False


def set_seed(seed):
    """ Method to set seed across runs to ensure reproducibility.
    It fixes seed for single-gpu machines.

    Args:
        seed (int): Seed to fix reproducibility. It should different for
            each run

    Returns:
        RandomState: fixed random state to initialize dataset iterators
    """
    torch.backends.cudnn.deterministic = True
    # set to false for reproducibility, True to boost performance
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    random_state = random.getstate()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return random_state


def split_dataset(dataset: torch.utils.data.Dataset, train_size: int,
                  val_size: int, seed: int) -> List[torch.utils.data.Dataset]:
    """ Function to split a dataset
    Args:
        dataset (torch.utils.data.Dataset): dataset to be splitted
        train_size (int): Number of samples to keep in training dataset
        val_size (int): Number of samples to keep in validation dataset
        seed (int): Seed to fix split for reproducibility
    Returns:
        List[torch.utils.data.Dataset]: The subsets of dataset chosen for
            training dataset and validation dataset
    """
    first_split = len(dataset) - val_size
    data_rest, val_dataset = torch.utils.data.random_split(
                        dataset, [first_split, val_size],
                        generator=torch.Generator().manual_seed(seed))
    size_train = min(len(data_rest), train_size)
    train_dataset, _ = torch.utils.data.random_split(
                        data_rest, [size_train, len(data_rest)-size_train],
                        generator=torch.Generator().manual_seed(seed))
    return train_dataset, val_dataset


if not args.gpus:
    args.precision = 32

if args.mlp_hidden != args.hidden*4:
    print(f"[INFO] In original paper, mlp_hidden(CURRENT:{args.mlp_hidden})"
          f"is set to: {args.hidden*4}(={args.hidden}*4)")


class Net(pl.LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()
        self.hparams.update(vars(hparams))
        self.model = get_model(hparams)
        self.criterion = get_criterion(args)
        if hparams.cutmix:
            self.cutmix = CutMix(hparams.size, beta=1.)
        if hparams.mixup:
            self.mixup = MixUp(alpha=1.)
        self.log_image_flag = hparams.api_key is None

    def forward(self, x, ss):
        return self.model(x, ss)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1,
                   self.hparams.beta2),
            weight_decay=self.hparams.weight_decay)
        self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.hparams.max_epochs,
            eta_min=self.hparams.min_lr)
        self.scheduler = warmup_scheduler.GradualWarmupScheduler(
            self.optimizer, multiplier=1.,
            total_epoch=self.hparams.warmup_epoch,
            after_scheduler=self.base_scheduler)
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):

        img, label = batch['data']
        memory_input, memory_label = batch['memory']
        if self.hparams.cutmix or self.hparams.mixup:
            if self.hparams.cutmix:
                img, label, rand_label, lambda_ = self.cutmix((img, label))
            elif self.hparams.mixup:
                if np.random.rand() <= 0.8:
                    img, label, rand_label, lambda_ = self.mixup((img, label))
                else:
                    img, label, rand_label, lambda_ = img, label, \
                        torch.zeros_like(label), 1.
            out = self.model(img, memory_input)
            loss = self.criterion(out, label)*lambda_ + \
                self.criterion(out, rand_label)*(1.-lambda_)
        else:
            out = self(img, memory_input)
            loss = self.criterion(out, label)

        if not self.log_image_flag and not self.hparams.dry_run:
            self.log_image_flag = True
            self._log_image(img.clone().detach().cpu())

        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("loss", loss)
        self.log("acc", acc)
        return loss

    def training_epoch_end(self, outputs):
        self.log("lr", self.optimizer.param_groups[0]["lr"])

    def validation_step(self, batch, batch_idx):
        img, label = batch['data']
        memory_input, memory_label = batch['memory']

        out = self(img, memory_input)
        loss = self.criterion(out, label)
        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

    def _log_image(self, image):
        grid = torchvision.utils.make_grid(image, nrow=4)
        self.logger.experiment.log_image(grid.permute(1, 2, 0))
        print("[INFO] LOG IMAGE!!!")


if __name__ == "__main__":
    experiment_name = get_experiment_name(args)
    print(experiment_name)
    if args.api_key:
        print("[INFO] Log with Comet.ml!")
        logger = pl.loggers.CometLogger(
            api_key=args.api_key,
            save_dir="logs",
            project_name=args.project_name,
            experiment_name=experiment_name
        )
        refresh_rate = 0
    else:
        print("[INFO] Log with CSV")
        logger = pl.loggers.CSVLogger(
            save_dir="logs",
            name=experiment_name
        )
        refresh_rate = 1
    accuracies = []
    for run in range(0, 15):
        train_ds, test_ds = get_dataset(args)
        train_data, _ = split_dataset(train_ds, args.data_samples, 1, run)
        set_seed(run)
        train_dl = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, pin_memory=True,
            shuffle=True, drop_last=True, worker_init_fn=run)

        test_dl = torch.utils.data.DataLoader(
            test_ds, batch_size=args.eval_batch_size,
            pin_memory=True, worker_init_fn=run)

        mem_loader = torch.utils.data.DataLoader(train_data, batch_size=100,
                                                 pin_memory=True, shuffle=True,
                                                 drop_last=True,
                                                 worker_init_fn=run)

        loaders = {"data": train_dl, "memory": mem_loader}
        combined_training_loaders = CombinedLoader(loaders, "max_size_cycle")
        test_loaders = {"data": test_dl, "memory": mem_loader}
        combined_test_loaders = CombinedLoader(test_loaders, "max_size_cycle")
        net = Net(args)
        trainer = pl.Trainer(precision=args.precision,
                             fast_dev_run=args.dry_run, gpus=args.gpus,
                             benchmark=args.benchmark, logger=logger,
                             max_epochs=args.max_epochs,
                             enable_model_summary=False,
                             progress_bar_refresh_rate=refresh_rate)
        trainer.fit(model=net, train_dataloaders=combined_training_loaders,
                    val_dataloaders=combined_test_loaders)
        print(trainer.callback_metrics['val_acc'])
        accuracies.append(trainer.callback_metrics['val_acc'])
        if not args.dry_run:
            model_path = f"weights/{experiment_name}.pth"
            torch.save(net.state_dict(), model_path)
            if args.api_key:
                logger.experiment.log_asset(file_name=experiment_name,
                                            file_data=model_path)
        print(f"Run:{run+1} | Mean Accuracy:{np.mean(accuracies):.4f} | "
              f"Std Dev Accuracy:{np.std(accuracies):.4f}\t")
