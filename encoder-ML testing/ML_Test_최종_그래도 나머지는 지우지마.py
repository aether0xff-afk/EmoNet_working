import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR


# =========================
# Utils
# =========================
def now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # rank -> pearson (scipy 없이)
    r_true = pd.Series(y_true).rank(method="average").to_numpy(dtype=float)
    r_pred = pd.Series(y_pred).rank(method="average").to_numpy(dtype=float)
    corr = float(np.corrcoef(r_true, r_pred)[0, 1])
    if np.isnan(corr):
        return 0.0
    return corr


def requires_dense(estimator: BaseEstimator) -> bool:
    # 희소행렬(sparse) 못 받는 모델들
    return isinstance(estimator, HistGradientBoostingRegressor)


class SafeTruncatedSVD(BaseEstimator, TransformerMixin):
    """
    TruncatedSVD는 n_components가 feature 수보다 크면 터질 수 있음.
    이 래퍼는 fit 시점에 안전하게 n_components를 조정해줌.
    """
    def __init__(self, n_components: int = 300, random_state: int = 42):
        self.n_components = int(n_components)
        self.random_state = int(random_state)
        self._svd: Optional[TruncatedSVD] = None
        self.used_n_components_: Optional[int] = None

    def fit(self, X, y=None):
        n_features = int(X.shape[1])
        # TruncatedSVD는 n_components <= n_features-1 권장(보수적으로)
        safe = min(self.n_components, max(1, n_features - 1))
        self.used_n_components_ = int(safe)
        self._svd = TruncatedSVD(n_components=self.used_n_components_, random_state=self.random_state)
        self._svd.fit(X)
        return self

    def transform(self, X):
        if self._svd is None:
            raise RuntimeError("SafeTruncatedSVD is not fitted.")
        return self._svd.transform(X)


def make_vectorizer(kind: str) -> TfidfVectorizer:
    # dtype=float32로 메모리 절약
    if kind == "word":
        return TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            dtype=np.float32,
        )
    if kind == "char":
        return TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.95,
            dtype=np.float32,
        )
    raise ValueError("vector_kind must be 'word' or 'char'")


def make_pipeline(
    model: BaseEstimator,
    vector_kind: str,
    use_svd: bool,
    svd_dim: int,
    random_state: int,
) -> Pipeline:
    steps: List[Tuple[str, Any]] = [("tfidf", make_vectorizer(vector_kind))]
    if use_svd:
        steps.append(("svd", SafeTruncatedSVD(n_components=svd_dim, random_state=random_state)))
    steps.append(("model", model))
    return Pipeline(steps)


@dataclass
class VectorCfg:
    name: str
    vector_kind: str
    use_svd: bool
    svd_dim: int = 300


def benchmark(
    df: pd.DataFrame,
    folds: int,
    random_state: int,
    vector_setups: List[VectorCfg],
    models: Dict[str, BaseEstimator],
    out_csv: Path,
) -> pd.DataFrame:
    X = df["text"].astype(str).to_numpy()
    y = df["y"].astype(float).to_numpy()

    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)

    rows: List[Dict[str, Any]] = []

    for vcfg in vector_setups:
        for mname, est in models.items():
            # ✅ dense 필요 모델은 SVD 없으면 무조건 스킵
            if requires_dense(est) and not vcfg.use_svd:
                print(f"[{vcfg.name} | {mname}] skipped (needs dense -> enable SVD)")
                rows.append({
                    "vector": vcfg.name,
                    "model": mname,
                    "status": "skipped_sparse",
                    "MAE(mean)": np.nan,
                    "RMSE(mean)": np.nan,
                    "R2(mean)": np.nan,
                    "Spearman(mean)": np.nan,
                    "fit_time_s(mean)": np.nan,
                    "pred_time_s(mean)": np.nan,
                    "n_samples": int(len(df)),
                    "error": "needs dense; sparse tfidf",
                })
                continue

            fit_times: List[float] = []
            pred_times: List[float] = []
            metrics: List[Tuple[float, float, float, float]] = []

            ok = True
            err_msg = ""

            fold_idx = 0
            for tr_idx, te_idx in kf.split(X):
                fold_idx += 1
                X_tr, X_te = X[tr_idx], X[te_idx]
                y_tr, y_te = y[tr_idx], y[te_idx]

                pipe = make_pipeline(
                    clone(est),
                    vector_kind=vcfg.vector_kind,
                    use_svd=vcfg.use_svd,
                    svd_dim=vcfg.svd_dim,
                    random_state=random_state,
                )

                try:
                    t0 = time.perf_counter()
                    pipe.fit(X_tr, y_tr)
                    t1 = time.perf_counter()

                    t2 = time.perf_counter()
                    y_pred = np.asarray(pipe.predict(X_te), dtype=float)  # 타입 안정
                    t3 = time.perf_counter()

                    fit_times.append(t1 - t0)
                    pred_times.append(t3 - t2)

                    mae = float(mean_absolute_error(y_te, y_pred))
                    mse = float(mean_squared_error(y_te, y_pred))  # squared 파라미터 없이
                    rmse = float(np.sqrt(mse))
                    r2 = float(r2_score(y_te, y_pred))
                    sp = float(spearman(y_te, y_pred))

                    metrics.append((mae, rmse, r2, sp))

                    print(
                        f"[{vcfg.name} | {mname}] fold {fold_idx}/{folds} "
                        f"MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f} Spearman={sp:.4f} "
                        f"fit={fit_times[-1]:.2f}s pred={pred_times[-1]:.2f}s"
                    )
                except Exception as e:
                    ok = False
                    err_msg = f"{type(e).__name__}: {str(e)}"
                    print(f"[{vcfg.name} | {mname}] ERROR on fold {fold_idx}/{folds} -> {err_msg}")
                    break

            if ok and len(metrics) > 0:
                m = np.array(metrics, dtype=float)
                row = {
                    "vector": vcfg.name,
                    "model": mname,
                    "status": "ok",
                    "MAE(mean)": float(np.nanmean(m[:, 0])),
                    "RMSE(mean)": float(np.nanmean(m[:, 1])),
                    "R2(mean)": float(np.nanmean(m[:, 2])),
                    "Spearman(mean)": float(np.nanmean(m[:, 3])),
                    "fit_time_s(mean)": float(np.mean(fit_times)),
                    "pred_time_s(mean)": float(np.mean(pred_times)),
                    "n_samples": int(len(df)),
                    "error": "",
                }
            else:
                row = {
                    "vector": vcfg.name,
                    "model": mname,
                    "status": "error",
                    "MAE(mean)": np.nan,
                    "RMSE(mean)": np.nan,
                    "R2(mean)": np.nan,
                    "Spearman(mean)": np.nan,
                    "fit_time_s(mean)": np.nan,
                    "pred_time_s(mean)": np.nan,
                    "n_samples": int(len(df)),
                    "error": err_msg,
                }

            rows.append(row)

            # 중간 저장(실험 오래 걸려도 결과 누적됨)
            pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

    res = pd.DataFrame(rows)

    # 정렬: ok인 것들만 위로
    ok_res = res[res["status"] == "ok"].copy()
    err_res = res[res["status"] != "ok"].copy()

    ok_res = ok_res.sort_values(["RMSE(mean)", "MAE(mean)"], ascending=[True, True])
    res = pd.concat([ok_res, err_res], ignore_index=True)

    res.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="out_benchmark/dataset_for_regression.csv")
    ap.add_argument("--outdir", type=str, default="out_benchmark")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--max_samples", type=int, default=30000)  # 0 또는 음수면 전체 사용
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--svd_dim", type=int, default=300)
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset csv not found: {dataset_path}")

    df = pd.read_csv(dataset_path)

    # 필수 컬럼 체크
    for col in ["text", "y", "label"]:
        if col not in df.columns:
            raise ValueError(f"'{col}' column not found in {dataset_path}")

    if args.max_samples > 0 and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)

    print("\n==============================")
    print("REGRESSION BENCHMARK START")
    print("==============================")
    print("python:", np.__version__)
    print("sklearn:", sklearn.__version__)
    print("samples:", len(df), "folds:", args.folds)

    vector_setups = [
        VectorCfg("word_tfidf", "word", False, args.svd_dim),
        VectorCfg("char_tfidf", "char", False, args.svd_dim),
        VectorCfg(f"word_tfidf_svd{args.svd_dim}", "word", True, args.svd_dim),
        VectorCfg(f"char_tfidf_svd{args.svd_dim}", "char", True, args.svd_dim),
    ]

    models: Dict[str, BaseEstimator] = {
        "Ridge": Ridge(alpha=2.0, random_state=args.seed),
        "ElasticNet": ElasticNet(alpha=1e-3, l1_ratio=0.3, random_state=args.seed, max_iter=5000),
        "LinearSVR": LinearSVR(C=1.0, random_state=args.seed, max_iter=5000),
        "HistGBDT": HistGradientBoostingRegressor(random_state=args.seed),
    }

    out_csv = out_dir / f"benchmark_results_{now_str()}.csv"
    result = benchmark(
        df=df,
        folds=args.folds,
        random_state=args.seed,
        vector_setups=vector_setups,
        models=models,
        out_csv=out_csv,
    )

    print("\n==============================")
    print("BENCHMARK DONE")
    print("==============================")
    # ok 결과만 보기 좋게 출력
    ok = result[result["status"] == "ok"].copy()
    if len(ok) > 0:
        print(ok.to_string(index=False))
        best = ok.iloc[0]
        print("\n--- BEST (by RMSE) ---")
        print(f"vector={best['vector']} model={best['model']}")
        print(f"RMSE={best['RMSE(mean)']:.4f} MAE={best['MAE(mean)']:.4f} R2={best['R2(mean)']:.4f} Spearman={best['Spearman(mean)']:.4f}")
        print(f"fit_time={best['fit_time_s(mean)']:.2f}s pred_time={best['pred_time_s(mean)']:.2f}s")
    else:
        print("No successful runs. Check error rows in the CSV.")

    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()