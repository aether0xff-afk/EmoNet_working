import pandas as pd
import numpy as np

path = "out_benchmark/label_map.csv"
df = pd.read_csv(path)

# 빈 y만 채움 (이미 채워진 값은 유지)
mask = df["y"].isna()

# 라벨을 정렬해서 0~1 균등 분포로 매핑 (재현성 좋음)
labels = df.loc[mask, "label"].astype(str).tolist()
labels_sorted = sorted(labels, key=lambda x: int(x[1:]) if x[1:].isdigit() else 10**9)

n = len(labels_sorted)
if n == 0:
    print("이미 y가 다 채워져 있어.")
else:
    ys = np.linspace(0.0, 1.0, n)
    mapping = dict(zip(labels_sorted, ys))

    df.loc[mask, "y"] = df.loc[mask, "label"].astype(str).map(mapping)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Filled y for {n} labels into {path}")
