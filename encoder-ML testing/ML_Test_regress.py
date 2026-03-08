import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.ensemble import HistGradientBoostingRegressor


# ==========================
# 설정
# ==========================
OUT_DIR = "out_benchmark"
DATASET_CSV = f"{OUT_DIR}/dataset_for_regression.csv"

FOLDS = 3                # 5만개면 5-fold는 꽤 무거움. 일단 3으로 추천
RANDOM_STATE = 42
MAX_SAMPLES = 30000      # 속도 너무 느리면 여기 줄이면 됨. None이면 전체 사용

# 벡터 설정: 일단 현실적인 2개만 (속도/성능 균형)
VECTOR_SETUPS = [
    {"name": "word_tfidf", "vector_kind": "word", "use_svd": False},
    {"name": "char_tfidf", "vector_kind": "char", "use_svd": False},
    {"name": "word_tfidf_svd300", "vector_kind": "word", "use_svd": True, "svd_dim": 300},
    {"name": "char_tfidf_svd300", "vector_kind": "char", "use_svd": True, "svd_dim": 300},
]

# 모델: 텍스트 회귀에서 속도/성능 괜찮은 애들 위주
MODELS = {
    "Ridge": Ridge(alpha=2.0, random_state=RANDOM_STATE),
    "ElasticNet": ElasticNet(alpha=1e-3, l1_ratio=0.3, random_state=RANDOM_STATE, max_iter=5000),
    "LinearSVR": LinearSVR(C=1.0, random_state=RANDOM_STATE),
    "HistGBDT": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
}


# ==========================
# Spearman (rank -> pearson)
# ==========================
def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r_true = pd.Series(y_true).rank(method="average").to_numpy(dtype=float)
    r_pred = pd.Series(y_pred).rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(r_true, r_pred)[0, 1])


def make_vectorizer(kind: str) -> TfidfVectorizer:
    if kind == "word":
        return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.95)
    if kind == "char":
        return TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_df=0.95)
    raise ValueError("vector_kind must be 'word' or 'char'")


def make_pipeline(model, vector_kind="word", use_svd=False, svd_dim=300) -> Pipeline:
    steps: List[Tuple[str, Any]] = [("tfidf", make_vectorizer(vector_kind))]
    if use_svd:
        steps.append(("svd", TruncatedSVD(n_components=svd_dim, random_state=RANDOM_STATE)))
    steps.append(("model", model))
    return Pipeline(steps)


def benchmark(df: pd.DataFrame, folds: int) -> pd.DataFrame:
    X = df["text"].astype(str).to_numpy()
    y = df["y"].astype(float).to_numpy()

    kf = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    rows: List[Dict[str, Any]] = []

    for vcfg in VECTOR_SETUPS:
        vname = vcfg["name"]
        vkind = vcfg["vector_kind"]
        use_svd = bool(vcfg["use_svd"])
        svd_dim = int(vcfg.get("svd_dim", 300))

        for mname, est in MODELS.items():
            fit_times: List[float] = []
            pred_times: List[float] = []
            metrics: List[Tuple[float, float, float, float]] = []

            fold_idx = 0
            for tr_idx, te_idx in kf.split(X):
                fold_idx += 1
                X_tr, X_te = X[tr_idx], X[te_idx]
                y_tr, y_te = y[tr_idx], y[te_idx]

                pipe = make_pipeline(clone(est), vector_kind=vkind, use_svd=use_svd, svd_dim=svd_dim)

                t0 = time.perf_counter()
                pipe.fit(X_tr, y_tr)
                t1 = time.perf_counter()

                t2 = time.perf_counter()
                y_pred = np.asarray(pipe.predict(X_te), dtype=float)
                t3 = time.perf_counter()

                fit_times.append(t1 - t0)
                pred_times.append(t3 - t2)

                mae = float(mean_absolute_error(y_te, y_pred))
                mse = float(mean_squared_error(y_te, y_pred))     # ✅ squared 인자 없이
                rmse = float(np.sqrt(mse))
                r2 = float(r2_score(y_te, y_pred))
                sp = float(spearman(y_te, y_pred))

                metrics.append((mae, rmse, r2, sp))

                print(
                    f"[{vname} | {mname}] fold {fold_idx}/{folds} "
                    f"MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f} Spearman={sp:.4f} "
                    f"fit={fit_times[-1]:.2f}s pred={pred_times[-1]:.2f}s"
                )

            m = np.array(metrics, dtype=float)
            rows.append({
                "vector": vname,
                "model": mname,
                "MAE(mean)": float(np.nanmean(m[:, 0])),
                "RMSE(mean)": float(np.nanmean(m[:, 1])),
                "R2(mean)": float(np.nanmean(m[:, 2])),
                "Spearman(mean)": float(np.nanmean(m[:, 3])),
                "fit_time_s(mean)": float(np.mean(fit_times)),
                "pred_time_s(mean)": float(np.mean(pred_times)),
                "n_samples": int(len(df)),
            })

    res = pd.DataFrame(rows).sort_values(["RMSE(mean)", "MAE(mean)"], ascending=[True, True]).reset_index(drop=True)
    return res


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(DATASET_CSV).exists():
        raise FileNotFoundError(
            f"{DATASET_CSV} 가 없어. 먼저 기존 ML_Test.py로 dataset_for_regression.csv 생성부터 해줘."
        )

    df = pd.read_csv(DATASET_CSV)

    # 안전 체크
    for col in ["text", "y", "label"]:
        if col not in df.columns:
            raise ValueError(f"{DATASET_CSV}에 '{col}' 컬럼이 없어.")

    # 샘플링(속도)
    if MAX_SAMPLES is not None and len(df) > MAX_SAMPLES:
        df = df.sample(n=MAX_SAMPLES, random_state=RANDOM_STATE).reset_index(drop=True)

    print("\n==============================")
    print("REGRESSION BENCHMARK START")
    print("==============================")
    print("samples:", len(df), "folds:", FOLDS)
    print("vector_setups:", [v["name"] for v in VECTOR_SETUPS])
    print("models:", list(MODELS.keys()))

    result = benchmark(df, folds=FOLDS)

    out_path = out_dir / "benchmark_results.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n==============================")
    print("BENCHMARK DONE (sorted by RMSE)")
    print("==============================")
    print(result.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()