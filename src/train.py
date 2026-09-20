"""ANN 학습 및 평가"""

from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn

from src.download_data import DATA_PATH
from src.model import LifeExpectancyANN

ROOT = Path(__file__).resolve().parents[1]
TARGET = "Life expectancy"
FEATURES = [
    "Adult Mortality", "infant deaths", "Alcohol", "percentage expenditure",
    "Hepatitis B", "BMI", "under-five deaths", "Polio", "Diphtheria",
    "HIV/AIDS", "GDP", "thinness  1-19 years",
    "Income composition of resources", "Schooling",
]
# 학습 설정
SEED = 42
HIDDEN_SIZES = (5, 10, 15)
LEARNING_RATE = 0.01
MAX_EPOCHS = 4000
PATIENCE = 300
MIN_DELTA = 1e-6


def load_data(path=DATA_PATH):
    """데이터 로딩"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Run python -m src.download_data first.")
    raw = pd.read_csv(path)
    raw.columns = raw.columns.str.strip()
    return raw


def clean_data(raw):
    """결측치 제거"""
    required = ["Country", "Year", TARGET, *FEATURES]
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")
    clean = raw[required].dropna(subset=[TARGET, *FEATURES]).copy()
    if not np.isfinite(clean[[TARGET, *FEATURES]].to_numpy()).all():
        raise ValueError("Model data contains infinite values.")
    if clean.duplicated(["Country", "Year"]).any():
        raise ValueError("Duplicate country-year observations need review before splitting.")
    return clean


def prepare_data(clean):
    """데이터 분할 및 정규화"""
    train_val, test = train_test_split(clean, test_size=0.2, random_state=SEED)
    train, val = train_test_split(train_val, test_size=0.25, random_state=SEED)
    frames = {"train": train, "validation": val, "test": test}
    x_scaler = MinMaxScaler().fit(train[FEATURES])
    y_scaler = MinMaxScaler().fit(train[[TARGET]])
    arrays = {}
    for name, frame in frames.items():
        arrays[name] = (
            torch.tensor(x_scaler.transform(frame[FEATURES]), dtype=torch.float32),
            torch.tensor(y_scaler.transform(frame[[TARGET]]), dtype=torch.float32),
        )
    return {
        "frames": frames, "arrays": arrays,
        "x_scaler": x_scaler, "y_scaler": y_scaler,
    }


def train_ann(data, hidden_nodes):
    """ANN 학습"""
    torch.manual_seed(SEED)
    model = LifeExpectancyANN(len(FEATURES), hidden_nodes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    x_train, y_train = data["arrays"]["train"]
    x_val, y_val = data["arrays"]["validation"]
    best_loss = float("inf")
    patience_reference = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        prediction = model(x_train)
        loss = loss_fn(prediction, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_loss = loss_fn(model(x_train), y_train).item()
            val_loss = loss_fn(model(x_val), y_val).item()
        if not np.isfinite([train_loss, val_loss]).all():
            raise RuntimeError(f"Non-finite loss for {hidden_nodes} nodes at epoch {epoch}.")
        history.append({
            "hidden_nodes": hidden_nodes, "epoch": epoch,
            "train_mse_scaled": train_loss, "validation_mse_scaled": val_loss,
        })
        # - 최저 검증 오차 가중치 저장
        # - min_delta 기준 중단 대기 횟수 계산
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = deepcopy(model.state_dict())
        if val_loss < patience_reference - MIN_DELTA:
            patience_reference = val_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= PATIENCE:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history), best_epoch


def predict_years(model, x, y_scaler):
    with torch.no_grad():
        scaled = model(x).numpy()
    return y_scaler.inverse_transform(scaled).ravel()


def evaluate(y_true, y_pred):
    """회귀 평가 지표"""
    return {
        "mae_years": float(mean_absolute_error(y_true, y_pred)),
        "rmse_years": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(np.corrcoef(y_true, y_pred)[0, 1]),
    }


def save_figures(histories, predictions, selected_nodes, best_epochs, output_dir):
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 130})
    fig, axes = plt.subplots(1, len(histories), figsize=(12, 3.8), sharey=True)
    for ax, (nodes, history) in zip(axes, histories.items()):
        ax.plot(history["epoch"], history["train_mse_scaled"], label="Train", color="#246a9b")
        ax.plot(history["epoch"], history["validation_mse_scaled"], label="Validation", color="#c56536")
        ax.axvline(best_epochs[nodes], color="#666666", linestyle=":", label="Best checkpoint")
        ax.set(title=f"{nodes} hidden nodes", xlabel="Epoch", yscale="log")
    axes[0].set_ylabel("MSE (scaled target, log axis)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Training and validation loss")
    fig.tight_layout()
    fig.savefig(figures / "learning_curves.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    actual = predictions["actual_years"]
    for ax, col, label, color in zip(
        axes, ["ann_prediction_years", "linear_prediction_years"],
        [f"ANN ({selected_nodes} hidden nodes)", "Linear regression"], ["#246a9b", "#c56536"],
    ):
        predicted = predictions[col]
        lo, hi = min(actual.min(), predicted.min()) - 2, max(actual.max(), predicted.max()) + 2
        ax.scatter(actual, predicted, s=15, alpha=0.55, color=color, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="#555555", label="Perfect prediction")
        ax.set(title=label, xlabel="Actual life expectancy (years)",
               ylabel="Predicted life expectancy (years)", xlim=(lo, hi), ylim=(lo, hi))
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=8)
    fig.suptitle("Held-out test observations")
    fig.tight_layout()
    fig.savefig(figures / "predictions.png", bbox_inches="tight")
    plt.close(fig)


def run_experiment(data=None, output_dir=ROOT / "results"):
    """모델 비교"""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if data is None:
        data = prepare_data(clean_data(load_data()))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models, histories, best_epochs = {}, {}, {}
    validation_rows = []
    for nodes in HIDDEN_SIZES:
        model, history, best_epoch = train_ann(data, nodes)
        models[nodes], histories[nodes], best_epochs[nodes] = model, history, best_epoch
        prediction = predict_years(model, data["arrays"]["validation"][0], data["y_scaler"])
        scores = evaluate(data["frames"]["validation"][TARGET], prediction)
        validation_rows.append({
            "hidden_nodes": nodes, "best_epoch": best_epoch, "epochs_run": len(history),
            **scores,
        })
    validation = pd.DataFrame(validation_rows)
    best_index = validation["rmse_years"].idxmin()
    selected_nodes = int(validation.loc[best_index, "hidden_nodes"])
    validation["selected"] = validation["hidden_nodes"].eq(selected_nodes)

    baseline = LinearRegression().fit(
        data["x_scaler"].transform(data["frames"]["train"][FEATURES]),
        data["frames"]["train"][TARGET],
    )
    test = data["frames"]["test"]
    ann_prediction = predict_years(models[selected_nodes], data["arrays"]["test"][0], data["y_scaler"])
    linear_prediction = baseline.predict(data["x_scaler"].transform(test[FEATURES]))
    metrics = pd.DataFrame([
        {"model": f"ANN ({selected_nodes} nodes)", "split": "test", **evaluate(test[TARGET], ann_prediction)},
        {"model": "Linear regression", "split": "test", **evaluate(test[TARGET], linear_prediction)},
    ])
    predictions = test[["Country", "Year"]].copy()
    predictions.insert(0, "source_row", test.index)
    predictions["actual_years"] = test[TARGET]
    predictions["ann_prediction_years"] = ann_prediction
    predictions["linear_prediction_years"] = linear_prediction
    split_rows = []
    for split, frame in data["frames"].items():
        rows = frame[["Country", "Year"]].copy()
        rows.insert(0, "source_row", frame.index)
        rows["split"] = split
        split_rows.append(rows)
    split_table = pd.concat(split_rows).sort_values("source_row")

    metrics.to_csv(output_dir / "metrics.csv", index=False)
    validation.to_csv(output_dir / "validation_results.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    split_table.to_csv(output_dir / "split.csv", index=False)
    pd.concat(histories.values()).to_csv(output_dir / "training_history.csv", index=False)
    save_figures(histories, predictions, selected_nodes, best_epochs, output_dir)
    return {"metrics": metrics, "validation": validation,
            "predictions": predictions, "histories": histories}


if __name__ == "__main__":
    result = run_experiment()
    print("Validation")
    print(result["validation"].round(4).to_string(index=False))
    print("\nTest")
    print(result["metrics"].round(4).to_string(index=False))
