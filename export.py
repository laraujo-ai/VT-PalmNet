"""
Export PalmNet embedding model to ONNX for edge deployment.

The exported model:
  Input  : (1, 1, 128, 128)  float32  — normalised grayscale palm ROI
  Output : (1, embed_dim)    float32  — L2-normalised embedding (default 256-dim)

The ArcFace classification head is NOT exported — only the embedding branch.
Usage
-----
# Export after training:
python palm_net/export.py \
    --weights ./palm_net/results/checkpoint/net_params_best.pth \
    --id_num  600 \
    --output  palm_net.onnx

# Verify with onnxruntime (requires: pip install onnxruntime):
python palm_net/export.py --weights ... --verify

Requirements
------------
pip install onnx onnxruntime   # for verification on the training machine
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model import PalmNet


class EmbeddingOnly(nn.Module):
    """Wraps PalmNet so that ONNX export captures only get_embedding()."""

    def __init__(self, palm_net: PalmNet):
        super().__init__()
        self.model = palm_net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.get_embedding(x)


def export_onnx(
    weights: str,
    id_num:  int,
    embed_dim: int,
    output:  str,
    opset:   int = 17,
) -> Path:
    out_path = Path(output)

    print(f"Loading weights from: {weights}")
    net = PalmNet(num_classes=id_num, embed_dim=embed_dim)
    state = torch.load(weights, map_location="cpu")
    net.load_state_dict(state)
    net.eval()

    wrapper = EmbeddingOnly(net)
    wrapper.eval()

    dummy = torch.randn(1, 1, 128, 128)

    print(f"Exporting to ONNX (opset {opset}) → {out_path}")
    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        opset_version=opset,
        input_names=["palm_roi"],
        output_names=["embedding"],
        dynamic_axes={
            "palm_roi":  {0: "batch"},
            "embedding": {0: "batch"},
        },
        export_params=True,
    )
    print(f"  Saved: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def verify_onnx(onnx_path: Path, weights: str, id_num: int, embed_dim: int):
    """Compare PyTorch and ONNX Runtime outputs on a random input."""
    try:
        import onnxruntime as ort
        import onnx
    except ImportError:
        print("\n[verify] onnx / onnxruntime not installed — skipping verification.")
        print("  Run: pip install onnx onnxruntime")
        return

    print("\nVerifying ONNX model …")

    net = PalmNet(num_classes=id_num, embed_dim=embed_dim)
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    wrapper = EmbeddingOnly(net)
    wrapper.eval()

    dummy = torch.randn(1, 1, 128, 128)
    with torch.no_grad():
        pt_out = wrapper(dummy).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(["embedding"], {"palm_roi": dummy.numpy()})[0]

    max_diff = np.abs(pt_out - ort_out).max()
    print(f"  PyTorch output shape : {pt_out.shape}")
    print(f"  ONNX RT output shape : {ort_out.shape}")
    print(f"  Max absolute diff    : {max_diff:.2e}")

    if max_diff < 1e-4:
        print("  PASSED — outputs match.")
    else:
        print("  WARNING — outputs differ more than expected.")

    dummy2 = torch.randn(1, 1, 128, 128)
    with torch.no_grad():
        emb1 = wrapper(dummy).numpy()[0]
        emb2 = wrapper(dummy2).numpy()[0]
    cos_sim = np.dot(emb1, emb2) 
    print(f"\nSanity: cosine sim between two random inputs = {cos_sim:.4f} (expect near 0)")

def parse_args():
    p = argparse.ArgumentParser(description="Export PalmNet to ONNX")
    p.add_argument("--weights",   required=True, help="Path to .pth weights file.")
    p.add_argument("--id_num",    type=int, default=600)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--output",    type=str, default="palm_net.onnx")
    p.add_argument("--opset",     type=int, default=17)
    p.add_argument("--verify",    action="store_true",
                   help="Verify ONNX output matches PyTorch (requires onnxruntime).")
    p.add_argument("--raspi",     action="store_true",
                   help="Print Raspberry Pi deployment instructions.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.raspi:
        print(RASPI_USAGE)
        return

    onnx_path = export_onnx(
        weights=args.weights,
        id_num=args.id_num,
        embed_dim=args.embed_dim,
        output=args.output,
        opset=args.opset,
    )

    if args.verify:
        verify_onnx(onnx_path, args.weights, args.id_num, args.embed_dim)

    print("\nDone.")


if __name__ == "__main__":
    main()
