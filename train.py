import os
import gc
import argparse
from distutils.util import strtobool
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from sklearn.model_selection import train_test_split

from utils import fix_seed
from utils import get_config
from logger import setup_logging
from trainer import Trainer, EarlyStopping
from models import DeepCrysTet
from data import Meshdataset, collate_batch


def main(args, save_dirs, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_regression = True if args.task == "regression" else False
    mesh_data = np.load(os.path.join(args.data_path))
    target_data = pd.read_csv(args.target_path, names=["mpid", "target"])
    dataset = Meshdataset(mesh_data, target_data, is_regression)
    feature_dim = dataset.feature_dim
    model = DeepCrysTet(feature_dim, is_regression).to(device)
    del mesh_data
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path)["model_state_dict"])
    batch_size = args.batch_size or cfg["batch_size"]
    if args.sampling_rate < 1.0:
        remove_rate = 1 - args.sampling_rate
        sampling_idx, _ = train_test_split(
            list(range(len(dataset))),
            test_size=remove_rate,
            random_state=cfg["seed"],
            shuffle=True,
        )
        dataset = Subset(dataset, sampling_idx)
    train_idx, valid_idx = train_test_split(
        list(range(len(dataset))), test_size=0.2, random_state=cfg["seed"], shuffle=True
    )
    train_dataset = Subset(dataset, train_idx)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=collate_batch,
        pin_memory=True,
    )
    dataset = Subset(dataset, valid_idx)
    valid_idx, test_idx = train_test_split(
        list(range(len(dataset))), test_size=0.5, random_state=cfg["seed"], shuffle=True
    )
    valid_dataset = Subset(dataset, valid_idx)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=collate_batch,
        pin_memory=True,
    )
    test_dataset = Subset(dataset, test_idx)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=collate_batch,
        pin_memory=True,
    )
    del dataset
    gc.collect()

    if is_regression:
        criterion = nn.L1Loss()
    else:
        criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    patience = args.es_patience or cfg["es_patience"]
    es = EarlyStopping(patience=patience, verbose=True)

    # Training
    trainer = Trainer(
        model,
        criterion,
        optimizer,
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        n_epochs=args.epochs,
        es=es,
        save_dirs=save_dirs,
        amp=strtobool(args.amp),
        feature_dim=feature_dim,
        is_regression=is_regression,
    )
    trainer.train()

    # Evaluation
    trainer.test()


if __name__ == "__main__":
    cfg = get_config()
    fix_seed(cfg["seed"])
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Input data path")
    parser.add_argument("--target-path", type=str, required=True, help="Target data path")
    parser.add_argument(
        "--save-dir", default="./saved", type=str, help="Save directory path (default: ./saved)"
    )
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs (default: 20)")
    parser.add_argument(
        "--model-path",
        default=None,
        type=str,
        help="Model path used for retraining (default: None)",
    )
    parser.add_argument(
        "--batch-size", default=0, type=int, help="The size of batch (default in config.yaml)"
    )
    parser.add_argument(
        "--es-patience",
        default=0,
        type=int,
        help="The number of patience epochs for EarlyStoppings (default in config.yaml)",
    )
    parser.add_argument(
        "--amp",
        default="False",
        type=str,
        help="Use Automatic Mixed Precision (default: False)",
    )
    parser.add_argument(
        "--run-id",
        default="",
        type=str,
        help="Run ID for saving results (default: '')",
    )
    parser.add_argument(
        "--task",
        choices=["regression", "classification"],
        default="regression",
        help="Select `regression` or `classification` task (default: regression)",
    )
    parser.add_argument(
        "--sampling-rate",
        default=1.0,
        type=float,
        help="Sampling rate of input data (default: 1.0)",
    )
    args = parser.parse_args()
    save_dir = args.save_dir
    run_id = args.run_id or datetime.now().strftime(r"%Y%m%d_%H%M%S")
    save_dirs = dict()
    save_dirs["model"] = os.path.join(save_dir, run_id, "models")
    save_dirs["log"] = os.path.join(save_dir, run_id, "logs")
    save_dirs["tensorboard"] = os.path.join(save_dir, run_id, "tensorboard")
    for dir in save_dirs.values():
        os.makedirs(dir, exist_ok=True)
    setup_logging(save_dirs["log"])

    main(args, save_dirs, cfg)
