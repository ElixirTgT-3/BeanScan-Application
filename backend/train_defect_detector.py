import torch
from pathlib import Path
import argparse

from ml.train_defect_detector import train


if __name__ == "__main__":
    # Resolve paths relative to repository root so it works from any CWD
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    data_root = (repo_root / "data").as_posix()
    output_dir = (repo_root / "models").as_posix()

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)  # optimized for RTX 3050
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    train(
        data_root=data_root,
        device_str="cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=output_dir,
        use_amp=False,
    )


