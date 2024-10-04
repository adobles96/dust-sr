from argparse import ArgumentParser
from pathlib import Path

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.utilities import rank_zero_info, rank_zero_warn
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split

from metrics import (
    get_extra_metrics,
    get_extra_metrics_on_inputs,
    get_metrics,
    get_metrics_on_inputs,
)
from dataset import TrainDataset
from utils.loss_functions import get_loss
from utils.file_handling import read_yaml_file
from models import Model

SEED = 42


def main(args) -> None:
    """Main function for training and testing the model.

    Parameters:
        args: Command line arguments.
    """
    seed_everything(SEED)
    cwd = Path.cwd()
    run_config: dict = read_yaml_file(cwd / f"{args.run_config}")
    dataset_config: dict = run_config["dataset"]
    model_config: dict = run_config["model"]
    model_config["batch_size"] = dataset_config["batch_size"]
    loss_config: dict = run_config["loss"]
    trainer_config: dict = run_config["trainer"]

    train, val = random_split(
        TrainDataset(
            args.train_patches,
            add_noise=dataset_config["add_noise"],
            noise_scale_factor=dataset_config["noise_scale_factor"]
        ),
        lengths=[0.8, 0.2]
    )
    train_loader = DataLoader(train, batch_size=dataset_config["batch_size"], shuffle=True,
                              num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val, batch_size=dataset_config["batch_size"], shuffle=False,
                            num_workers=4, persistent_workers=True)
    rank_zero_info("Data loaded.")

    loss = get_loss(loss_config)
    rank_zero_info(f"Using loss function: {loss}")

    prefix = "val" if args.stage == "train" else "test"
    metrics = get_metrics(prefix=prefix) if 1 == 0 else None  # TODO: Fix this -- George
    metrics_on_inputs = (
        get_metrics_on_inputs(prefix=prefix) if 1 == 0 else None
    )  # TODO: Fix this -- George
    extra_metrics = get_extra_metrics(prefix=prefix) if args.stage == "test" else None
    extra_metrics_on_inputs = (
        get_extra_metrics_on_inputs(prefix=prefix) if args.stage == "test" else None
    )

    if trainer_config["overfit_batches"] > 0:
        total_steps = trainer_config["overfit_batches"] * trainer_config["epochs"]
    else:
        total_steps = len(train) * trainer_config["epochs"]

    model = Model(
        model_config,
        loss=loss,
        metrics=metrics,
        metrics_on_inputs=metrics_on_inputs,
        extra_metrics=extra_metrics,
        extra_metrics_on_inputs=extra_metrics_on_inputs,
        total_steps=total_steps,
    )

    callbacks = None

    if args.stage == "train":
        callbacks = []
        model_checkpoint = ModelCheckpoint(
            monitor="val/loss",
            dirpath="checkpoints",
            filename=f"epoch:{{epoch:05d}}-val_loss:{{val/loss:.5f}}",  # noqa
            mode="min",
            auto_insert_metric_name=False,
        )
        callbacks.append(model_checkpoint)

    trainer = Trainer(
        accelerator=trainer_config["accelerator"],
        devices=trainer_config["devices"] if args.stage == "train" else 1,
        max_epochs=trainer_config["epochs"],
        strategy=trainer_config["strategy"],
        callbacks=callbacks,
        benchmark=True,  # speeds up training if input is not changing size
        overfit_batches=trainer_config["overfit_batches"],
        log_every_n_steps=trainer_config["log_every_n_steps"],
    )

    if trainer_config["checkpoint_path"] is not None:
        rank_zero_warn("A checkpoint_path is being used.")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=trainer_config["checkpoint_path"],
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--stage", choices=["train", "test"], default="train", help="What stage to execute",
    )
    parser.add_argument(
        "--run_config", type=Path, default="config.yml", help="Yaml file name of the run config.",
    )
    parser.add_argument("--train_patches", type=Path, default="data/training_patches.pkl")
    args = parser.parse_args()

    main(args)
