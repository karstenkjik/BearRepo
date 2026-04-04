from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import torch

from model import LidarBinaryCNN2D


def safe_load_checkpoint(checkpoint_path: Path, device: torch.device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)
    except Exception:
        return torch.load(checkpoint_path, map_location=device)


def load_processed_dataset(split_root: Path):
    files = sorted(split_root.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found under {split_root}")

    xs = []
    ys = []
    file_paths = []

    for file_path in files:
        with np.load(file_path) as data:
            x = data["x"].astype(np.float32)   # expected (2, T, 240)
            y = int(data["y"])
            xs.append(x)
            ys.append(y)
            file_paths.append(str(file_path))

    xs = np.stack(xs, axis=0)   # (N, 2, T, 240)
    ys = np.array(ys, dtype=np.int64)
    return file_paths, xs, ys


def infer_probabilities(model, xs: np.ndarray, device: torch.device):
    x_tensor = torch.from_numpy(xs).to(device)

    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1)

    return probs.cpu().numpy()  # shape (N, 2)


def compute_metrics(ys: np.ndarray, horse_probs: np.ndarray, threshold: float):
    preds = (horse_probs >= threshold).astype(np.int64)

    tp = int(np.sum((preds == 1) & (ys == 1)))
    tn = int(np.sum((preds == 0) & (ys == 0)))
    fp = int(np.sum((preds == 1) & (ys == 0)))
    fn = int(np.sum((preds == 0) & (ys == 1)))

    total = max(1, len(ys))
    accuracy = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return {
        "threshold": float(threshold),
        "num_samples": int(len(ys)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
        "predictions": preds,
    }


def threshold_sweep(ys: np.ndarray, horse_probs: np.ndarray):
    results = []
    best_by_accuracy = None
    best_by_f1 = None

    for threshold in np.linspace(0.05, 0.95, 19):
        metrics = compute_metrics(ys, horse_probs, float(threshold))
        results.append(metrics)

        if best_by_accuracy is None or metrics["accuracy"] > best_by_accuracy["accuracy"]:
            best_by_accuracy = metrics

        if best_by_f1 is None or metrics["f1"] > best_by_f1["f1"]:
            best_by_f1 = metrics

    return results, best_by_accuracy, best_by_f1


def main():
    # CHANGE THESE TWO PATHS
    processed_test_root = Path(
        "/Users/karstenkempfe/Downloads/NewLidar-trainingModel/Training2D/processed_temporal_binary_t5s2/test"
    )
    checkpoint_path = Path(
        "/Users/karstenkempfe/Downloads/NewLidar-trainingModel/Training2D/bestmodel.pt"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_paths, xs, ys = load_processed_dataset(processed_test_root)

    checkpoint = safe_load_checkpoint(checkpoint_path, device)
    model = LidarBinaryCNN2D().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    probs = infer_probabilities(model, xs, device)
    horse_probs = probs[:, 1]

    print(f"Loaded {len(ys)} processed test samples")
    print(f"x shape: {xs.shape}")
    print(f"Positive samples: {int(np.sum(ys == 1))}")
    print(f"Negative samples: {int(np.sum(ys == 0))}")

    # Argmax-equivalent threshold for binary softmax isn't exactly 0.5 in all contexts,
    # but 0.5 is the cleanest thresholded baseline.
    baseline = compute_metrics(ys, horse_probs, threshold=0.5)
    print("\nBaseline @ threshold=0.50")
    print(json.dumps({k: v for k, v in baseline.items() if k != "predictions"}, indent=2))

    sweep, best_acc, best_f1 = threshold_sweep(ys, horse_probs)

    print("\nBest by accuracy")
    print(json.dumps({k: v for k, v in best_acc.items() if k != "predictions"}, indent=2))

    print("\nBest by F1")
    print(json.dumps({k: v for k, v in best_f1.items() if k != "predictions"}, indent=2))

    print("\nThreshold sweep summary")
    for row in sweep:
        print(
            f"thr={row['threshold']:.2f} | "
            f"acc={row['accuracy']:.4f} | "
            f"prec={row['precision']:.4f} | "
            f"rec={row['recall']:.4f} | "
            f"f1={row['f1']:.4f} | "
            f"tp={row['confusion_matrix']['tp']} "
            f"tn={row['confusion_matrix']['tn']} "
            f"fp={row['confusion_matrix']['fp']} "
            f"fn={row['confusion_matrix']['fn']}"
        )

    # Optional: show a few sample predictions
    print("\nFirst 15 example predictions")
    preds = best_f1["predictions"]
    for i in range(min(15, len(file_paths))):
        print(
            f"{file_paths[i]} | y={ys[i]} | horse_prob={horse_probs[i]:.6f} | pred={int(preds[i])}"
        )


if __name__ == "__main__":
    main()