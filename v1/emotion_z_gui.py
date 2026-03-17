#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emotion Z GUI (integrated folder mode)
- experiment folder 하나만 고르면 dataset / benchmark / label_map 자동 탐색
- emotion_z_pipeline.py의 GUI 래퍼
- train / infer / H 시각화
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import queue
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False


HERE = Path(__file__).resolve().parent
PIPELINE_PATH = HERE / "emotion_z_pipeline.py"


def load_pipeline_module():
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(f"emotion_z_pipeline.py not found: {PIPELINE_PATH}")
    spec = importlib.util.spec_from_file_location("emotion_z_pipeline", PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load emotion_z_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["emotion_z_pipeline"] = module
    spec.loader.exec_module(module)
    return module


class QueueWriter(io.TextIOBase):
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def write(self, s: str) -> int:
        if s:
            self.q.put(("log", s))
        return len(s)

    def flush(self) -> None:
        return None


class EmotionZGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Emotion Z Pipeline GUI (통합형)")
        self.geometry("1360x900")
        self.minsize(1200, 780)

        self.log_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.pipeline_module = None
        self.loaded_pipeline = None
        self.last_result: Optional[Dict[str, Any]] = None

        self._build_vars()
        self._build_ui()
        self._try_load_module()
        self.after(120, self._poll_queue)

    def _build_vars(self) -> None:
        cwd = Path.cwd()
        self.experiment_dir = tk.StringVar(value=str(cwd).replace("\\", "/"))
        self.dataset_csv = tk.StringVar(value="")
        self.benchmark_csv = tk.StringVar(value="")
        self.label_map_csv = tk.StringVar(value="")
        self.model_out = tk.StringVar(value=str(cwd / "emotion_z_pipeline.pt").replace("\\", "/"))
        self.model_path = tk.StringVar(value=str(cwd / "emotion_z_pipeline.pt").replace("\\", "/"))

        self.seed = tk.IntVar(value=42)
        self.stimulus_max_samples = tk.IntVar(value=30000)
        self.history_train_samples = tk.IntVar(value=512)
        self.future_horizon = tk.IntVar(value=3)
        self.min_prefix_len = tk.IntVar(value=4)
        self.prefix_stride = tk.IntVar(value=2)
        self.z_dim = tk.IntVar(value=16)
        self.hidden_dim = tk.IntVar(value=64)
        self.batch_size = tk.IntVar(value=32)
        self.lr = tk.DoubleVar(value=1e-3)
        self.epochs = tk.IntVar(value=5)
        self.alpha_future = tk.DoubleVar(value=0.5)
        self.device = tk.StringVar(value="auto")

        self.status_text = tk.StringVar(value="준비됨")
        self.module_status_text = tk.StringVar(value="로딩 중...")
        self.discovery_text = tk.StringVar(value="아직 자동 탐색 전")
        self.input_text = tk.StringVar(value="왜 이렇게 일이 많지 너무 지친다")

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        header = ttk.Frame(root)
        header.pack(fill="x")
        ttk.Label(header, text="Emotion Z Pipeline GUI (통합형)", font=("맑은 고딕", 16, "bold")).pack(side="left")
        ttk.Label(header, textvariable=self.module_status_text, foreground="#555").pack(side="left", padx=12)
        ttk.Label(header, textvariable=self.status_text, foreground="#0a5").pack(side="right")

        main = ttk.Panedwindow(root, orient="horizontal")
        main.pack(fill="both", expand=True, pady=(10, 0))

        left = ttk.Frame(main, padding=6)
        right = ttk.Frame(main, padding=6)
        main.add(left, weight=0)
        main.add(right, weight=1)

        self._build_left_panel(left)
        self._build_right_panel(right)

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        notebook = ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True)

        train_tab = ttk.Frame(notebook, padding=8)
        infer_tab = ttk.Frame(notebook, padding=8)
        notebook.add(train_tab, text="학습")
        notebook.add(infer_tab, text="추론")

        folder_box = ttk.LabelFrame(train_tab, text="실험 폴더(통합 입력)", padding=8)
        folder_box.pack(fill="x", pady=(0, 8))
        row = ttk.Frame(folder_box)
        row.pack(fill="x")
        ttk.Label(row, text="experiment_dir", width=18).pack(side="left")
        ttk.Entry(row, textvariable=self.experiment_dir).pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(row, text="폴더 선택", command=self._browse_folder).pack(side="left")
        ttk.Button(row, text="자동 탐색", command=self.autodiscover_files).pack(side="left", padx=(6, 0))

        ttk.Label(folder_box, textvariable=self.discovery_text, foreground="#555", wraplength=430, justify="left").pack(fill="x", pady=(8, 0))

        files_box = ttk.LabelFrame(train_tab, text="자동 인식된 파일", padding=8)
        files_box.pack(fill="x", pady=(0, 8))
        self._readonly_row(files_box, "dataset_csv", self.dataset_csv)
        self._readonly_row(files_box, "benchmark_csv", self.benchmark_csv)
        self._readonly_row(files_box, "label_map_csv", self.label_map_csv)
        self._path_row(files_box, "model_out", self.model_out, "save")

        cfg_box = ttk.LabelFrame(train_tab, text="학습 설정", padding=8)
        cfg_box.pack(fill="x", pady=(0, 8))
        fields = [
            ("seed", self.seed),
            ("stimulus_max_samples", self.stimulus_max_samples),
            ("history_train_samples", self.history_train_samples),
            ("future_horizon", self.future_horizon),
            ("min_prefix_len", self.min_prefix_len),
            ("prefix_stride", self.prefix_stride),
            ("z_dim", self.z_dim),
            ("hidden_dim", self.hidden_dim),
            ("batch_size", self.batch_size),
            ("lr", self.lr),
            ("epochs", self.epochs),
            ("alpha_future", self.alpha_future),
            ("device", self.device),
        ]
        for label, var in fields:
            row = ttk.Frame(cfg_box)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=22).pack(side="left")
            ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)

        btns = ttk.Frame(train_tab)
        btns.pack(fill="x", pady=(4, 0))
        ttk.Button(btns, text="학습 시작", command=self.start_train).pack(side="left", fill="x", expand=True)
        ttk.Button(btns, text="로그 지우기", command=self.clear_log).pack(side="left", fill="x", expand=True, padx=(6, 0))

        infer_files = ttk.LabelFrame(infer_tab, text="모델", padding=8)
        infer_files.pack(fill="x", pady=(0, 8))
        self._path_row(infer_files, "model_path", self.model_path, "model")

        text_box = ttk.LabelFrame(infer_tab, text="입력 텍스트", padding=8)
        text_box.pack(fill="both", expand=False, pady=(0, 8))
        self.text_widget = tk.Text(text_box, height=10, wrap="word")
        self.text_widget.pack(fill="both", expand=True)
        self.text_widget.insert("1.0", self.input_text.get())

        infer_btns = ttk.Frame(infer_tab)
        infer_btns.pack(fill="x")
        ttk.Button(infer_btns, text="모델 불러오기", command=self.load_model).pack(side="left", fill="x", expand=True)
        ttk.Button(infer_btns, text="z 추론", command=self.run_infer).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(infer_btns, text="결과 JSON 저장", command=self.save_result_json).pack(side="left", fill="x", expand=True)

        log_box = ttk.LabelFrame(parent, text="로그", padding=6)
        log_box.pack(fill="both", expand=True, pady=(8, 0))
        self.log_text = tk.Text(log_box, height=16, wrap="word")
        self.log_text.pack(fill="both", expand=True)

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent)
        top.pack(fill="both", expand=True)

        result_box = ttk.LabelFrame(top, text="추론 결과", padding=8)
        result_box.pack(fill="x")
        self.result_text = tk.Text(result_box, height=18, wrap="word")
        self.result_text.pack(fill="both", expand=True)

        plot_box = ttk.LabelFrame(top, text="히스토리 H 시각화", padding=8)
        plot_box.pack(fill="both", expand=True, pady=(8, 0))

        if MATPLOTLIB_OK:
            self.figure = Figure(figsize=(7.5, 5.5), dpi=100)
            self.ax1 = self.figure.add_subplot(211)
            self.ax2 = self.figure.add_subplot(212)
            self.canvas = FigureCanvasTkAgg(self.figure, master=plot_box)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
        else:
            self.figure = None
            self.ax1 = None
            self.ax2 = None
            self.canvas = None
            ttk.Label(plot_box, text="matplotlib을 불러오지 못해서 그래프를 표시할 수 없음").pack(fill="both", expand=True)

    def _normalize_path_text(self, value: str) -> str:
        value = str(value).strip().strip('"\'')
        if not value:
            return value
        value = value.replace("\\", "/")
        return os.path.normpath(value).replace("\\", "/")

    def _readonly_row(self, parent: ttk.Frame, label: str, var: tk.StringVar) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=16).pack(side="left")
        entry = ttk.Entry(row, textvariable=var, state="readonly")
        entry.pack(side="left", fill="x", expand=True)

    def _path_row(self, parent: ttk.Frame, label: str, var: tk.StringVar, mode: str) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=16).pack(side="left")
        ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(row, text="찾기", command=lambda v=var, m=mode: self._browse(v, m)).pack(side="left")

    def _browse(self, var: tk.StringVar, mode: str) -> None:
        if mode == "save":
            path = filedialog.asksaveasfilename(defaultextension=".pt", filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")])
        elif mode == "model":
            path = filedialog.askopenfilename(filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")])
        else:
            path = filedialog.askopenfilename(filetypes=[("All files", "*.*")])
        if path:
            var.set(self._normalize_path_text(path))

    def _browse_folder(self) -> None:
        path = filedialog.askdirectory(initialdir=self._normalize_path_text(self.experiment_dir.get()) or str(Path.cwd()))
        if path:
            norm = self._normalize_path_text(path)
            self.experiment_dir.set(norm)
            self._suggest_model_out_from_folder(norm)
            self.autodiscover_files()

    def _suggest_model_out_from_folder(self, folder: str) -> None:
        folder_path = Path(folder)
        self.model_out.set(str((folder_path / "emotion_z_pipeline.pt")).replace("\\", "/"))
        if not self.model_path.get().strip() or self.model_path.get().endswith("emotion_z_pipeline.pt"):
            self.model_path.set(self.model_out.get())

    def _try_load_module(self) -> None:
        try:
            self.pipeline_module = load_pipeline_module()
            self.module_status_text.set(f"모듈 로드 완료: {PIPELINE_PATH.name}")
            self.append_log(f"[OK] pipeline module loaded: {PIPELINE_PATH}\n")
        except Exception as e:
            self.pipeline_module = None
            self.module_status_text.set("모듈 로드 실패")
            self.append_log(f"[ERROR] failed to load module: {e}\n")

    def append_log(self, text: str) -> None:
        self.log_text.insert("end", text)
        self.log_text.see("end")

    def clear_log(self) -> None:
        self.log_text.delete("1.0", "end")

    def set_status(self, text: str) -> None:
        self.status_text.set(text)
        self.update_idletasks()

    def _run_in_thread(self, func, *args, **kwargs) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showwarning("실행 중", "이미 작업이 실행 중이야. 끝난 뒤 다시 해줘.")
            return
        self.worker_thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        self.worker_thread.start()

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "log":
                    self.append_log(str(payload))
                elif kind == "status":
                    self.set_status(str(payload))
                elif kind == "train_done":
                    self.model_path.set(self.model_out.get())
                    self.set_status("학습 완료")
                    messagebox.showinfo("완료", f"모델 저장 완료\n{payload}")
                elif kind == "model_loaded":
                    self.set_status("모델 로드 완료")
                    messagebox.showinfo("완료", f"모델을 불러왔어.\n{payload}")
                elif kind == "infer_done":
                    self.last_result = payload
                    self.set_status("추론 완료")
                    self._display_result(payload)
                elif kind == "error":
                    self.set_status("오류 발생")
                    messagebox.showerror("오류", str(payload))
        except queue.Empty:
            pass
        self.after(120, self._poll_queue)

    # ---------- integrated discovery ----------
    def _csv_has_columns(self, path: Path, required: Iterable[str]) -> bool:
        try:
            df = pd.read_csv(path, nrows=3)
            cols = set(df.columns.astype(str).tolist())
            return all(col in cols for col in required)
        except Exception:
            return False

    def _candidate_key(self, root: Path, p: Path) -> tuple[int, int, str]:
        try:
            rel_parts = len(p.relative_to(root).parts)
        except Exception:
            rel_parts = 9999
        return (rel_parts, len(str(p)), str(p).lower())

    def _pick_first_valid(self, root: Path, patterns: list[str], required_cols: Iterable[str]) -> Optional[Path]:
        candidates: list[Path] = []
        for pattern in patterns:
            candidates.extend(root.rglob(pattern))
        unique: list[Path] = []
        seen = set()
        for c in candidates:
            s = str(c.resolve()).lower()
            if s not in seen and c.is_file():
                seen.add(s)
                unique.append(c)
        unique.sort(key=lambda p: self._candidate_key(root, p))
        for c in unique:
            if self._csv_has_columns(c, required_cols):
                return c
        return None

    def autodiscover_files(self) -> None:
        folder = self._normalize_path_text(self.experiment_dir.get())
        if not folder:
            messagebox.showwarning("경고", "먼저 실험 폴더를 지정해줘.")
            return
        root = Path(folder)
        if not root.exists() or not root.is_dir():
            messagebox.showerror("오류", f"폴더를 찾지 못했어.\n{folder}")
            return

        dataset = self._pick_first_valid(root, ["dataset_for_regression.csv", "*dataset_for_regression*.csv"], ["text", "y"])
        benchmark = self._pick_first_valid(root, ["benchmark_results_*.csv", "*benchmark*.csv"], ["vector", "model"])
        label_map = self._pick_first_valid(root, ["label_map.csv", "*label_map*.csv"], ["label", "y"])

        self.dataset_csv.set(str(dataset).replace("\\", "/") if dataset else "")
        self.benchmark_csv.set(str(benchmark).replace("\\", "/") if benchmark else "")
        self.label_map_csv.set(str(label_map).replace("\\", "/") if label_map else "")
        self._suggest_model_out_from_folder(folder)

        parts = [f"탐색 폴더: {folder}"]
        parts.append(f"dataset: {'OK' if dataset else '없음'}")
        parts.append(f"benchmark: {'OK' if benchmark else '없음(없어도 학습 가능)'}")
        parts.append(f"label_map: {'OK' if label_map else '없음(없어도 학습 가능)'}")
        self.discovery_text.set(" | ".join(parts))

        if dataset:
            self.append_log(f"[AutoDetect] dataset_csv = {dataset}\n")
        if benchmark:
            self.append_log(f"[AutoDetect] benchmark_csv = {benchmark}\n")
        if label_map:
            self.append_log(f"[AutoDetect] label_map_csv = {label_map}\n")
        if not dataset:
            self.append_log("[AutoDetect] dataset_for_regression.csv를 찾지 못했어.\n")

    def _validate_train_inputs(self) -> None:
        dataset_path = Path(self._normalize_path_text(self.dataset_csv.get())) if self.dataset_csv.get().strip() else None
        if dataset_path is None or not dataset_path.exists():
            raise FileNotFoundError("dataset_for_regression.csv를 찾지 못했어. 자동 탐색을 누르거나 폴더를 다시 골라줘.")
        if not self._csv_has_columns(dataset_path, ["text", "y"]):
            raise ValueError(f"dataset_csv 형식이 아니야. text, y 열이 필요해.\n{dataset_path}")

        benchmark_path = Path(self._normalize_path_text(self.benchmark_csv.get())) if self.benchmark_csv.get().strip() else None
        if benchmark_path and benchmark_path.exists() and not self._csv_has_columns(benchmark_path, ["vector", "model"]):
            raise ValueError(f"benchmark_csv 형식이 아니야. vector, model 열이 필요해.\n{benchmark_path}")

        label_map_path = Path(self._normalize_path_text(self.label_map_csv.get())) if self.label_map_csv.get().strip() else None
        if label_map_path and label_map_path.exists() and not self._csv_has_columns(label_map_path, ["label", "y"]):
            raise ValueError(f"label_map_csv 형식이 아니야. label, y 열이 필요해.\n{label_map_path}")

    # ---------- actions ----------
    def start_train(self) -> None:
        if self.pipeline_module is None:
            messagebox.showerror("오류", "emotion_z_pipeline.py를 불러오지 못했어.")
            return
        try:
            self._validate_train_inputs()
        except Exception as e:
            messagebox.showerror("입력 오류", str(e))
            return
        self._run_in_thread(self._train_worker)

    def _train_worker(self) -> None:
        qwriter = QueueWriter(self.log_queue)
        try:
            self.log_queue.put(("status", "학습 중..."))
            module = self.pipeline_module
            device = self.device.get().strip() or "auto"
            if device == "auto":
                device = "cuda" if module.torch.cuda.is_available() else "cpu"

            cfg = module.PipelineTrainConfig(
                seed=int(self.seed.get()),
                stimulus_max_samples=int(self.stimulus_max_samples.get()),
                history_train_samples=int(self.history_train_samples.get()),
                future_horizon=int(self.future_horizon.get()),
                min_prefix_len=int(self.min_prefix_len.get()),
                prefix_stride=int(self.prefix_stride.get()),
                z_dim=int(self.z_dim.get()),
                hidden_dim=int(self.hidden_dim.get()),
                batch_size=int(self.batch_size.get()),
                lr=float(self.lr.get()),
                epochs=int(self.epochs.get()),
                alpha_future=float(self.alpha_future.get()),
                device=device,
            )

            dataset_path = Path(self._normalize_path_text(self.dataset_csv.get()))
            benchmark_txt = self._normalize_path_text(self.benchmark_csv.get())
            labelmap_txt = self._normalize_path_text(self.label_map_csv.get())
            model_out = Path(self._normalize_path_text(self.model_out.get()))
            model_out.parent.mkdir(parents=True, exist_ok=True)

            pipeline = module.EmotionZPipeline(seed=int(self.seed.get()))
            with contextlib.redirect_stdout(qwriter), contextlib.redirect_stderr(qwriter):
                meta = pipeline.fit(
                    dataset_csv=dataset_path,
                    benchmark_csv=Path(benchmark_txt) if benchmark_txt else None,
                    label_map_csv=Path(labelmap_txt) if labelmap_txt else None,
                    config=cfg,
                )
                pipeline.save(model_out)
            self.loaded_pipeline = pipeline
            self.log_queue.put(("log", "\n[Saved] " + str(model_out).replace("\\", "/") + "\n"))
            self.log_queue.put(("log", json.dumps(meta, ensure_ascii=False, indent=2) + "\n"))
            self.log_queue.put(("train_done", str(model_out).replace("\\", "/")))
        except Exception:
            self.log_queue.put(("error", traceback.format_exc()))

    def load_model(self) -> None:
        if self.pipeline_module is None:
            messagebox.showerror("오류", "emotion_z_pipeline.py를 불러오지 못했어.")
            return
        self._run_in_thread(self._load_model_worker)

    def _load_model_worker(self) -> None:
        try:
            self.log_queue.put(("status", "모델 로드 중..."))
            module = self.pipeline_module
            model_path = self._normalize_path_text(self.model_path.get())
            pipeline = module.EmotionZPipeline.load(Path(model_path), map_location="cpu")
            self.loaded_pipeline = pipeline
            self.log_queue.put(("log", f"[Loaded] {model_path}\n"))
            self.log_queue.put(("model_loaded", model_path))
        except Exception:
            self.log_queue.put(("error", traceback.format_exc()))

    def run_infer(self) -> None:
        if self.pipeline_module is None:
            messagebox.showerror("오류", "emotion_z_pipeline.py를 불러오지 못했어.")
            return
        self._run_in_thread(self._infer_worker)

    def _infer_worker(self) -> None:
        try:
            self.log_queue.put(("status", "추론 중..."))
            text = self.text_widget.get("1.0", "end").strip()
            if not text:
                raise ValueError("입력 텍스트가 비어 있어.")
            if self.loaded_pipeline is None:
                module = self.pipeline_module
                self.loaded_pipeline = module.EmotionZPipeline.load(Path(self._normalize_path_text(self.model_path.get())), map_location="cpu")
            result = self.loaded_pipeline.encode_text(text)
            self.log_queue.put(("infer_done", result))
        except Exception:
            self.log_queue.put(("error", traceback.format_exc()))

    def _display_result(self, result: Dict[str, Any]) -> None:
        H = np.asarray(result["H"], dtype=np.float32)
        z = [float(v) for v in result["z"]]
        final_emo = [float(v) for v in result["final_emo"]]
        app = result["appraisal"]

        lines = []
        lines.append(f"text: {result['text']}")
        lines.append(f"score: {float(result['score']):.6f}")
        lines.append(f"H_shape: {result['H_shape']}")
        lines.append("")
        lines.append("u (appraisal):")
        lines.append("  " + ", ".join(f"{v:.4f}" for v in result["u"]))
        lines.append("h (dopamine, serotonin, norepinephrine, melatonin):")
        lines.append("  " + ", ".join(f"{v:.4f}" for v in result["h"]))
        lines.append("")
        lines.append("appraisal named:")
        for k, v in app.items():
            lines.append(f"  - {k}: {float(v):.4f}")
        lines.append("")
        lines.append("z:")
        lines.append("  " + ", ".join(f"{v:.6f}" for v in z))
        lines.append("final_emo:")
        lines.append("  " + ", ".join(f"{v:.4f}" for v in final_emo))

        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", "\n".join(lines))

        if self.ax1 is not None and H.size > 0:
            self.ax1.clear()
            self.ax2.clear()
            t = np.arange(H.shape[0])

            emo = H[:, :4]
            self.ax1.plot(t, emo[:, 0], label="dopamine")
            self.ax1.plot(t, emo[:, 1], label="serotonin")
            self.ax1.plot(t, emo[:, 2], label="norepinephrine")
            self.ax1.plot(t, emo[:, 3], label="melatonin")
            self.ax1.set_title("H 감정 궤적 (앞 4차원)")
            self.ax1.legend(loc="best")
            self.ax1.grid(True, alpha=0.3)

            if H.shape[1] >= 8:
                self.ax2.plot(t, H[:, 4], label="firing_rate")
                self.ax2.plot(t, H[:, 5], label="avg_threshold")
                self.ax2.plot(t, H[:, 6], label="avg_memory_size")
                self.ax2.plot(t, H[:, 7], label="rewires_add")
                self.ax2.set_title("H 네트워크 요약")
                self.ax2.legend(loc="best")
                self.ax2.grid(True, alpha=0.3)

            self.figure.tight_layout()
            self.canvas.draw_idle()

    def save_result_json(self) -> None:
        if not self.last_result:
            messagebox.showwarning("경고", "저장할 추론 결과가 없어.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if not path:
            return
        payload = dict(self.last_result)
        payload["H"] = np.asarray(payload["H"], dtype=float).tolist()
        payload["z"] = np.asarray(payload["z"], dtype=float).tolist()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        messagebox.showinfo("완료", f"결과를 저장했어.\n{path}")


def main() -> None:
    app = EmotionZGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
