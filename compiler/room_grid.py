"""Simplified RoomGrid for turbovec-integration-ccc.

Minimal adaptive forward engine with explicit config-change tracking
so that HotSwapIntegration can detect weight mutations (breed/rebirth)
and trigger automatic recompilation.
"""
from __future__ import annotations

__all__ = ["RoomGrid", "forward_einsum", "batch_novelty", "make_weights"]

import math
import logging
import sys
from typing import Any, Dict, Optional

import numpy as np

log = logging.getLogger(__name__)

# ── Numba availability ────────────────────────────────────
try:
    from numba import njit
    import numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    njit = None  # type: ignore[misc,assignment]


def make_weights(n: int, d: int = 64, h: int = 32, l: int = 16, seed: int = 42):
    """Deep 64→h→l MLP weights."""
    rng = np.random.RandomState(seed)
    w3 = np.eye(l, dtype=np.float32) * 0.99
    w3 += rng.randn(l, l).astype(np.float32) * 0.001
    return {
        "w1": rng.randn(n, d, h).astype(np.float32) * 0.01,
        "b1": np.zeros((1, n, h), dtype=np.float32),
        "w2": rng.randn(n, h, l).astype(np.float32) * 0.01,
        "b2": np.zeros((1, n, l), dtype=np.float32),
        "w3": np.broadcast_to(w3, (n, l, l)).copy(),
        "b3": np.zeros((1, n, l), dtype=np.float32),
    }


def forward_einsum(w, x):
    """Numpy einsum fallback: (n, l) latents."""
    x = x.ravel().astype(np.float32)
    h = np.einsum("d,ndh->nh", x, w["w1"], optimize=False) + w["b1"][0]
    h = np.maximum(h, 0, out=h)
    h = np.einsum("nh,nhl->nl", h, w["w2"], optimize=False) + w["b2"][0]
    h = np.maximum(h, 0, out=h)
    return np.einsum("nl,nll->nl", h, w["w3"], optimize=False) + w["b3"][0]


def _batch_novelty_numpy(latents, hist, hist_count, hist_idx, hist_max):
    """Pure numpy novelty implementation (fallback)."""
    n = latents.shape[0]
    norms = np.linalg.norm(latents, axis=1, keepdims=True) + 1e-8
    zn = latents / norms

    offsets = [(hist_idx - 1) % hist_max,
               (hist_idx - 2) % hist_max,
               (hist_idx - 3) % hist_max]
    hist_tensor = hist[offsets].transpose(1, 0, 2)

    hist_mask = np.zeros((n, 3), dtype=np.float32)
    for j in range(3):
        hist_mask[:, j] = (hist_count >= j + 1).astype(np.float32)

    h_norms = np.linalg.norm(hist_tensor, axis=-1, keepdims=True) + 1e-8
    hn = hist_tensor / h_norms
    sims = (zn[:, np.newaxis, :] * hn).sum(axis=-1)

    mask_sum = hist_mask.sum(axis=1, keepdims=True) + 1e-8
    mean_sim = (sims * hist_mask).sum(axis=1, keepdims=True) / mask_sum
    novelty = 1.0 - mean_sim.ravel()

    no_hist = hist_mask.sum(axis=1) < 2
    novelty[no_hist] = 0.5
    return novelty


if _HAS_NUMBA:
    @njit(cache=True, fastmath=True)  # type: ignore[misc]
    def _batch_novelty_numba_inner(latents, h1, h2, h3, hist_count):
        """Numba-compiled novelty kernel."""
        n = latents.shape[0]
        l = latents.shape[1]

        zn = np.empty((n, l), dtype=np.float32)
        for i in range(n):
            norm_sq = 0.0
            for j in range(l):
                v = latents[i, j]
                norm_sq += v * v
            norm = np.sqrt(norm_sq) + 1e-8
            for j in range(l):
                zn[i, j] = latents[i, j] / norm

        sims = np.empty((n, 3), dtype=np.float32)
        for i in range(n):
            for k, h in enumerate((h1, h2, h3)):
                norm_sq = 0.0
                for j in range(l):
                    v = h[i, j]
                    norm_sq += v * v
                norm = np.sqrt(norm_sq) + 1e-8
                sim = 0.0
                for j in range(l):
                    sim += zn[i, j] * (h[i, j] / norm)
                sims[i, k] = sim

        novelty = np.empty(n, dtype=np.float32)
        for i in range(n):
            count = hist_count[i]
            if count < 2:
                novelty[i] = 0.5
            else:
                total = 0.0
                valid = 0
                for k in range(3):
                    if k < count:
                        total += sims[i, k]
                        valid += 1
                if valid > 0:
                    mean_sim = total / valid
                    novelty[i] = 1.0 - mean_sim
                else:
                    novelty[i] = 0.5
        return novelty

    def _batch_novelty_numba(latents, hist, hist_count, hist_idx, hist_max):
        """Python wrapper: extracts ring buffer slices, calls Numba kernel."""
        h1 = hist[(hist_idx - 1) % hist_max]
        h2 = hist[(hist_idx - 2) % hist_max]
        h3 = hist[(hist_idx - 3) % hist_max]
        return _batch_novelty_numba_inner(latents, h1, h2, h3, hist_count)

    batch_novelty = _batch_novelty_numba
else:
    batch_novelty = _batch_novelty_numpy


class RoomGrid:
    """N rooms × MLP. Forward only. No training.

    Config-change tracking:
        Every operation that mutates weights (breed, rebirth) bumps
        ``_config_version``. HotSwapIntegration watches this counter
        to trigger automatic recompilation.
    """

    def __init__(self, n=250, d=64, h=32, l=16, chaos=0.3,
                 compiler=None):
        self.n = n
        self.w = make_weights(n, d, h, l)
        self.activity = np.zeros(n, dtype=np.int32)
        self.chaos = np.full(n, chaos, dtype=np.float32)
        self.ticks = 0
        self.l = l
        self.latents = np.zeros((n, l), dtype=np.float32)
        self._hist_max = 20
        self._hist = np.zeros((self._hist_max, n, l), dtype=np.float32)
        self._hist_idx = 0
        self._hist_count = np.zeros(n, dtype=np.int32)

        # ── Config-change tracking ─────────────────────────
        self._config_version = 0
        self._last_compile_version = 0

        # ── Compiler integration ───────────────────────────
        self._compiler = None
        if compiler is not None:
            if compiler == "auto":
                try:
                    from compiler.compiler import RoomGridCompiler
                    self._compiler = RoomGridCompiler(self)
                    self._compiler.auto_compile(ticks=50, ab_trials=30)
                except Exception as e:
                    log.warning("RoomGrid auto-compile failed: %s", e)
            elif hasattr(compiler, "auto_compile"):
                self._compiler = compiler
                compiler.grid = self
            else:
                raise TypeError("compiler must be 'auto' or a RoomGridCompiler instance")

    def _bump_config_version(self):
        """Increment version whenever weights or topology change."""
        self._config_version += 1

    def _forward(self, x):
        return forward_einsum(self.w, x)

    def tick(self, x):
        self.ticks += 1
        latents = self._forward(x)
        self.latents = latents
        self._hist[self._hist_idx] = latents
        self._hist_idx = (self._hist_idx + 1) % self._hist_max
        self._hist_count = np.minimum(self._hist_count + 1, self._hist_max)

        # ── Compiled routing fast-path ─────────────────────
        _tick_routing_compiled = getattr(sys.modules.get("compiler.room_grid"), "_tick_routing_compiled", None)
        if _tick_routing_compiled is not None:
            nv = batch_novelty(latents, self._hist, self._hist_count,
                               self._hist_idx, self._hist_max)
            fired_mask, new_chaos, fired_count = _tick_routing_compiled(
                latents, self.chaos, self.n,
                self._hist, self._hist_count, self._hist_idx, self._hist_max,
            )
            self.chaos = new_chaos
            fired = np.where(fired_mask)[0].tolist()[:10]
            self.activity[fired_mask] += 1
            return {"fired": fired_count, "ids": fired, "tick": self.ticks}
        # ── Fallback vectorised novelty + chaos gating ───────
        nv = batch_novelty(latents, self._hist, self._hist_count,
                           self._hist_idx, self._hist_max)
        chaos_fire = np.random.random(self.n) < self.chaos
        fired_mask = (nv > 0.5) | chaos_fire
        fired = np.where(fired_mask)[0].tolist()[:10]
        self.activity[fired_mask] += 1
        self.chaos = np.where(fired_mask,
                              np.maximum(0.01, self.chaos * 0.99),
                              self.chaos)
        return {"fired": int(fired_mask.sum()), "ids": fired, "tick": self.ticks}

    def rebirth(self, i):
        """Reset room `i` to random weights."""
        rng = np.random.RandomState(i + 9999)
        for k, shp in [("w1", (64, 32)), ("w2", (32, 16)), ("w3", (16, 16))]:
            self.w[k][i] = rng.randn(*shp).astype(np.float32) * 0.01
        self.activity[i] = 0
        self.chaos[i] = 0.3
        self._hist[:, i, :] = 0.0
        self._hist_count[i] = 0
        self._bump_config_version()

    def breed(self, src, dst):
        """Clone src weights to dst + mutation."""
        for k in ("w1", "w2", "w3"):
            self.w[k][dst] = self.w[k][src].copy()
        rng = np.random.RandomState(dst + 8888)
        for k in ("w1", "w2", "w3"):
            self.w[k][dst] += rng.randn(*self.w[k][dst].shape).astype(np.float32) * 0.005
        self.activity[dst] = 0
        self.chaos[dst] = 0.3
        self._hist[:, dst, :] = 0.0
        self._hist_count[dst] = 0
        self._bump_config_version()

    def cold(self, thresh=1):
        return [int(i) for i in range(self.n) if self.activity[i] < thresh]

    def top(self, k=10):
        idx = np.argsort(self.activity)[::-1][:k]
        return [(int(i), int(self.activity[i])) for i in idx]

    def __repr__(self):
        return (f"RoomGrid(n={self.n}, ticks={self.ticks}, "
                f"active={int((self.activity > 0).sum())}, "
                f"config_v={self._config_version})")
