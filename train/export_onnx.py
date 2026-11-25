"""Export a fine-tuned cross-encoder to ONNX for fastembed, optionally quantized to int8, and check it.

The output directory has the layout fastembed expects (config and tokenizer files at the top,
onnx/model.onnx), so the server can load it with TextCrossEncoder(..., specific_model_path=dir) or, once
uploaded to Hugging Face, by name. The check scores the same pairs with PyTorch and with onnxruntime
and fails if they differ, so the model that is shipped is the model that was evaluated.

    .venv-train/bin/python train/export_onnx.py train/models/l6-mix train/onnx/l6-mix [--int8]
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PAIRS = [
    (
        "two-loop corrections to the decay were first computed in",
        "Two-loop QCD corrections to B decays. We compute ...",
    ),
    ("paper that measured the W boson mass at the Tevatron", "Measurement of the W mass. Using data collected ..."),
    ("entanglement of phonon pairs in a quasi one dimensional Bose gas", "Entangled phonons. We study a Bose gas ..."),
]


def export(src: Path, out: Path, int8: bool) -> Path:
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_pretrained(src).eval()
    out.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(out)
    model.config.save_pretrained(out)
    batch = tok([q for q, _ in PAIRS], [d for _, d in PAIRS], padding=True, truncation=True, return_tensors="pt")
    names = [n for n in ("input_ids", "attention_mask", "token_type_ids") if n in batch]
    onnx_dir = out / "onnx"
    onnx_dir.mkdir(exist_ok=True)
    path = onnx_dir / "model.onnx"
    torch.onnx.export(
        model,
        tuple(batch[n] for n in names),
        str(path),
        input_names=names,
        output_names=["logits"],
        dynamic_axes={**{n: {0: "batch", 1: "sequence"} for n in names}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    if int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantized = onnx_dir / "model_int8.onnx"
        quantize_dynamic(str(path), str(quantized), weight_type=QuantType.QInt8)
        shutil.move(quantized, path)
    return path


def check(src: Path, path: Path, int8: bool) -> None:
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_pretrained(src).eval()
    batch = tok([q for q, _ in PAIRS], [d for _, d in PAIRS], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        ref = model(**batch).logits[:, 0].numpy()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {i.name: batch[i.name].numpy() for i in sess.get_inputs()}
    got = sess.run(None, feed)[0][:, 0]
    # int8 moves scores a little: whether it keeps the ranking is measured on the development pools
    # (eval/rerank_lab.py with the exported directory), not on three pairs
    tolerance = 0.6 if int8 else 1e-3
    print(f"pytorch {np.round(ref, 3)}  onnx {np.round(got, 3)}")
    if np.abs(ref - got).max() > tolerance or (not int8 and (np.argsort(ref) != np.argsort(got)).any()):
        raise SystemExit("ONNX scores differ from PyTorch")
    start = time.perf_counter()
    docs = [PAIRS[0][1] * 8] * 50  # 50 abstract-length documents, as the server scores
    enc = tok([PAIRS[0][0]] * 50, docs, padding=True, truncation=True, max_length=320, return_tensors="np")
    sess.run(None, {i.name: enc[i.name] for i in sess.get_inputs()})
    print(f"onnxruntime CPU: {time.perf_counter() - start:.2f}s for 50 candidates (indicative, this desktop)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--int8", action="store_true")
    args = parser.parse_args()
    path = export(args.src, args.out, args.int8)
    check(args.src, path, args.int8)
    print(f"ONNX model in {args.out}")


if __name__ == "__main__":
    main()
