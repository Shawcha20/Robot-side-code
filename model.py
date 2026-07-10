#!/usr/bin/env python3
"""
Ishara — model + inference (Phase 3c-4).

Extracted verbatim from robot_phase4.py: the SignTransformer stack, the label-file
lookup (name / ID2NAME), the 5-seed ensemble load, and predict(). predict() lived
in the features section before but belongs with the models it runs, so it moves
here; nothing about it changes. Public names: name, predict.

robot_phase4.py now does `from model import name, predict` in place of the MODEL
block, and removes the old predict() from the features section. The 5 models still
load at import time (same "loaded 5 models | N classes" message), the same 5-seed
softmax averaging is used, and DEVICE / FEATURE_DIM / CKPT / WORD_XLSX come from config.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import openpyxl

from config import CKPT, WORD_XLSX, DEVICE, FEATURE_DIM


class SpatialEncoder(nn.Module):
    def __init__(self, d_in, d=128, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(p),
            nn.Linear(d, d), nn.LayerNorm(d), nn.GELU())
    def forward(self, x): return self.net(x)


class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__(); self.q = nn.Linear(d, 1)
    def forward(self, x):
        w = torch.softmax(self.q(x).squeeze(-1), 1).unsqueeze(-1)
        return (x * w).sum(1)


class SignTransformer(nn.Module):
    def __init__(self, d_in, n, d=128, heads=4, layers=4, ff=4, p=0.3):
        super().__init__()
        self.encoder = SpatialEncoder(d_in, d, p)
        self.pos = nn.Parameter(torch.zeros(1, 512, d))
        layer = nn.TransformerEncoderLayer(d, heads, d * ff, p,
                                           batch_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, layers)
        self.pool = AttnPool(d)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(p), nn.Linear(d, n))
    def forward(self, x):
        t = self.encoder(x); t = t + self.pos[:, :t.shape[1]]
        return self.head(self.pool(self.tf(t)))


ws = openpyxl.load_workbook(WORD_XLSX).active
ID2NAME = {int(r[0]): str(r[1]).strip()
           for r in ws.iter_rows(min_row=2, values_only=True) if r[1] is not None}
def name(i): return ID2NAME.get(int(i), "class_" + str(i))

files = ["word_model_full.pt", "word_model_128_s2.pt", "word_model_128_s3.pt",
         "word_model_128_s4.pt", "word_model_128_s5.pt"]
first = torch.load(os.path.join(CKPT, files[0]), map_location=DEVICE, weights_only=True)
N_CLASSES = first["head.2.weight"].shape[0]
MODELS = []
for fn in files:
    m = SignTransformer(FEATURE_DIM, N_CLASSES).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(CKPT, fn), map_location=DEVICE, weights_only=True))
    m.eval(); MODELS.append(m)
print("loaded", len(MODELS), "models |", N_CLASSES, "classes")


@torch.no_grad()
def predict(buf):
    x = torch.from_numpy(np.stack(buf)[None]).float()
    p = sum(torch.softmax(m(x), 1) for m in MODELS) / len(MODELS)
    p = p[0].numpy()
    return int(p.argmax()), float(p.max())
