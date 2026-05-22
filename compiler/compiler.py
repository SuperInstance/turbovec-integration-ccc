"""RoomGridCompiler — auto-compiles RoomGrid hot paths and hot-swaps at runtime.

This is a self-contained copy of the compiler logic adapted for
`turbovec-integration-ccc`.  It profiles `grid.tick()`, compiles the
slowest phase with Numba, A/B tests for correctness + speedup, and
hot-swaps the module-level function if the compiled kernel wins.
"""
from __future__ import annotations

__all__ = ["RoomGridCompiler", "CompileResult"]

import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

try:
    from numba import njit
    import numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    njit = None  # type: ignore[misc,assignment]


# ── Numba kernels (defined at import so cache works) ──────

if _HAS_NUMBA:
    @njit(cache=True, fastmath=True)  # type: ignore[misc]
    def _forward_einsum_numba_core(
        x: np.ndarray,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
        w3: np.ndarray,
        b3: np.ndarray,
        n: int,
    ) -> np.ndarray:
        """Serial MLP forward: x(64) → (n, 16)."""
        out = np.empty((n, 16), dtype=np.float32)
        for i in range(n):
            h = np.empty(32, dtype=np.float32)
            for j in range(32):
                s = 0.0
                for k in range(64):
                    s += x[k] * w1[i, k, j]
                h[j] = s + b1[i, j]
            for j in range(32):
                if h[j] < 0.0:
                    h[j] = 0.0
            h2 = np.empty(16, dtype=np.float32)
            for j in range(16):
                s = 0.0
                for k in range(32):
                    s += h[k] * w2[i, k, j]
                h2[j] = s + b2[i, j]
            for j in range(16):
                if h2[j] < 0.0:
                    h2[j] = 0.0
            for j in range(16):
                s = 0.0
                for k in range(16):
                    s += h2[k] * w3[i, k, j]
                out[i, j] = s + b3[i, j]
        return out

    @njit(cache=True, fastmath=True)  # type: ignore[misc]
    def _routing_numba_core(
        nv: np.ndarray,
        chaos: np.ndarray,
        n: int,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Numba chaos/fire decision kernel."""
        fired_mask = np.empty(n, dtype=np.bool_)
        new_chaos = np.empty(n, dtype=np.float32)
        fired_count = 0
        for i in range(n):
            chaos_fire = np.random.random() < chaos[i]
            fire = (nv[i] > 0.5) or chaos_fire
            fired_mask[i] = fire
            if fire:
                new_chaos[i] = max(0.01, chaos[i] * 0.99)
                fired_count += 1
            else:
                new_chaos[i] = chaos[i]
        return fired_mask, new_chaos, fired_count
else:
    _forward_einsum_numba_core = None  # type: ignore[assignment]
    _routing_numba_core = None       # type: ignore[assignment]


@dataclass
class CompileResult:
    """Result of compiling a single RoomGrid hot path."""
    target: str
    original: Callable
    compiled: Callable
    speedup: float
    validated: bool
    hot_swapped: bool
    error: Optional[str] = None
    profile_ms: Dict[str, float] = field(default_factory=dict)


class RoomGridCompiler:
    """Auto-compiler for RoomGrid hot paths.

    Steps:
        1. Profile `grid.tick()` — time each sub-phase
        2. Pick the slowest phase as *hot path*
        3. Compile to Numba (or skip if unavailable)
        4. A/B test for correctness + speedup
        5. Hot-swap if `speedup > 1.0` **and** outputs match
        6. `restore_original()` rolls everything back
    """

    def __init__(self, grid: Any) -> None:
        self.grid = grid
        self._originals: Dict[str, Callable] = {}
        self.results: Dict[str, CompileResult] = {}
        self._has_numba = _HAS_NUMBA
        log.info("RoomGridCompiler initialised (n=%d, numba=%s)", grid.n, self._has_numba)

    # ── Profiling ───────────────────────────────────────────

    def profile_tick(self, ticks: int = 100,
                     signal: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Profile `grid.tick()` and return per-phase timings."""
        if signal is None:
            np.random.seed(42)
            signal = np.random.randn(64).astype(np.float32)

        forward_ms = 0.0
        novelty_ms = 0.0
        routing_ms = 0.0

        for _ in range(ticks):
            t0 = time.perf_counter()
            latents = self.grid._forward(signal)
            forward_ms += (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            from compiler.room_grid import batch_novelty
            _ = batch_novelty(latents, self.grid._hist, self.grid._hist_count,
                              self.grid._hist_idx, self.grid._hist_max)
            novelty_ms += (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            chaos_fire = np.random.random(self.grid.n) < self.grid.chaos
            _ = (np.zeros(self.grid.n, dtype=np.float32) > 0.5) | chaos_fire
            routing_ms += (time.perf_counter() - t0) * 1000

        totals = {
            "forward": forward_ms / ticks,
            "novelty": novelty_ms / ticks,
            "routing": routing_ms / ticks,
        }
        log.info("Profile (%d ticks): %s", ticks, totals)
        return totals

    # ── A/B testing ─────────────────────────────────────────

    def _ab_test(
        self,
        original: Callable,
        compiled: Callable,
        args_gen: Callable[[], Tuple],
        trials: int = 50,
        rtol: float = 1e-3,
        atol: float = 1e-5,
    ) -> Tuple[bool, float]:
        """A/B test: correctness + speedup.

        Returns ``(validated, speedup)``.
        """
        # 1. Correctness — 5 quick trials with different random seeds
        validated = True
        for seed in range(5):
            np.random.seed(seed)
            args = args_gen()
            try:
                expected = original(*args)
                actual = compiled(*args)
            except Exception as e:
                log.warning("A/B correctness error: %s", e)
                validated = False
                break
            if not self._outputs_equal(expected, actual, rtol=rtol, atol=atol):
                validated = False
                break

        if not validated:
            return False, 1.0

        # 2. Speedup — ``trials`` iterations
        np.random.seed(42)
        args = args_gen()
        t0 = time.perf_counter()
        for _ in range(trials):
            original(*args)
        t_orig = (time.perf_counter() - t0) * 1000

        np.random.seed(42)
        args = args_gen()
        t0 = time.perf_counter()
        for _ in range(trials):
            compiled(*args)
        t_comp = (time.perf_counter() - t0) * 1000

        speedup = t_orig / max(t_comp, 0.001)
        log.info("A/B speedup: %.2f×  (orig %.3f ms  comp %.3f ms)",
                 speedup, t_orig, t_comp)
        return validated, speedup

    @staticmethod
    def _outputs_equal(a: Any, b: Any, rtol: float = 1e-3,
                       atol: float = 1e-5) -> bool:
        if type(a) != type(b):
            return False
        if isinstance(a, np.ndarray):
            return np.allclose(a, b, rtol=rtol, atol=atol)
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all(
                RoomGridCompiler._outputs_equal(x, y, rtol=rtol, atol=atol)
                for x, y in zip(a, b)
            )
        return a == b

    # ── Hot-swap plumbing ───────────────────────────────────

    def _hot_swap(self, module_name: str, attr_name: str,
                  replacement: Callable, key: str) -> bool:
        """Replace a module-level function and keep the original for rollback."""
        mod = sys.modules.get(module_name)
        if mod is None:
            try:
                mod = __import__(module_name, fromlist=["_"])
            except ImportError:
                return False

        original = getattr(mod, attr_name, None)
        self._originals[key] = original

        if hasattr(original, "__qualname__"):
            replacement.__qualname__ = original.__qualname__
        if hasattr(original, "__name__"):
            replacement.__name__ = original.__name__
        replacement.__doc__ = getattr(original, "__doc__", None)
        replacement._sunset_original = original  # type: ignore[attr-defined]

        setattr(mod, attr_name, replacement)
        return True

    def _restore(self, key: str) -> bool:
        """Restore one hot-swapped function."""
        if key not in self._originals:
            return False
        original = self._originals.pop(key)
        mod_name, attr_name = key.rsplit(".", 1)
        mod = sys.modules.get(mod_name)
        if mod is None:
            return False
        if original is None:
            if hasattr(mod, attr_name):
                delattr(mod, attr_name)
        else:
            setattr(mod, attr_name, original)
        return True

    # ── Individual compilations ─────────────────────────────

    def compile_einsum(self, ab_trials: int = 50) -> CompileResult:
        """Compile `compiler.room_grid.forward_einsum` to Numba."""
        if not self._has_numba or _forward_einsum_numba_core is None:
            return CompileResult(
                target="einsum",
                original=self._noop,
                compiled=self._noop,
                speedup=1.0,
                validated=False,
                hot_swapped=False,
                error="Numba not available",
            )

        from compiler.room_grid import forward_einsum

        def _compiled_forward(w, x):
            xflat = x.ravel().astype(np.float32)
            n = w["w1"].shape[0]
            b1 = w["b1"][0] if w["b1"].ndim == 3 else w["b1"]
            b2 = w["b2"][0] if w["b2"].ndim == 3 else w["b2"]
            b3 = w["b3"][0] if w["b3"].ndim == 3 else w["b3"]
            return _forward_einsum_numba_core(
                xflat,
                w["w1"], b1,
                w["w2"], b2,
                w["w3"], b3,
                n,
            )

        def _gen_args():
            np.random.seed(42)
            x = np.random.randn(64).astype(np.float32)
            return (self.grid.w, x)

        validated, speedup = self._ab_test(
            forward_einsum, _compiled_forward, _gen_args, trials=ab_trials,
            rtol=1e-3, atol=1e-4,
        )

        result = CompileResult(
            target="einsum",
            original=forward_einsum,
            compiled=_compiled_forward,
            speedup=speedup,
            validated=validated,
            hot_swapped=False,
        )

        if validated and speedup > 1.0:
            swapped = self._hot_swap(
                "compiler.room_grid", "forward_einsum",
                _compiled_forward, "compiler.room_grid.forward_einsum",
            )
            result.hot_swapped = swapped
            if swapped:
                log.info("🔥 Hot-swapped forward_einsum — %.2f× speedup", speedup)

        self.results["einsum"] = result
        return result

    def compile_routing(self, ab_trials: int = 50) -> CompileResult:
        """Compile the chaos / fire / activity routing kernel to Numba."""
        if not self._has_numba or _routing_numba_core is None:
            return CompileResult(
                target="routing",
                original=self._noop,
                compiled=self._noop,
                speedup=1.0,
                validated=False,
                hot_swapped=False,
                error="Numba not available",
            )

        from compiler.room_grid import batch_novelty

        def _compiled_routing(latents, chaos, n, hist, hist_count, hist_idx, hist_max):
            nv = batch_novelty(latents, hist, hist_count, hist_idx, hist_max)
            return _routing_numba_core(nv, chaos, n)

        def _original_routing(latents, chaos, n, hist, hist_count, hist_idx, hist_max):
            nv = batch_novelty(latents, hist, hist_count, hist_idx, hist_max)
            chaos_fire = np.random.random(n) < chaos
            fired_mask = (nv > 0.5) | chaos_fire
            new_chaos = np.where(
                fired_mask,
                np.maximum(0.01, chaos * 0.99),
                chaos,
            )
            fired_count = int(fired_mask.sum())
            return fired_mask, new_chaos, fired_count

        def _gen_args():
            np.random.seed(42)
            n = self.grid.n
            latents = np.random.randn(n, 16).astype(np.float32)
            chaos = np.full(n, 0.3, dtype=np.float32)
            hist = np.zeros((self.grid._hist_max, n, 16), dtype=np.float32)
            hist_count = np.full(n, 5, dtype=np.int32)
            return (latents, chaos, n, hist, hist_count, self.grid._hist_idx, self.grid._hist_max)

        validated, speedup = self._ab_test(
            _original_routing, _compiled_routing, _gen_args,
            trials=ab_trials, rtol=1e-3, atol=1e-4,
        )

        result = CompileResult(
            target="routing",
            original=_original_routing,
            compiled=_compiled_routing,
            speedup=speedup,
            validated=validated,
            hot_swapped=False,
        )

        if validated and speedup > 1.0:
            swapped = self._hot_swap(
                "compiler.room_grid", "_tick_routing_compiled",
                _compiled_routing, "compiler.room_grid._tick_routing_compiled",
            )
            result.hot_swapped = swapped
            if swapped:
                log.info("🔥 Hot-swapped routing kernel — %.2f× speedup", speedup)

        self.results["routing"] = result
        return result

    # ── Auto-compile (profile → compile hottest) ───────────

    def auto_compile(self, ticks: int = 100, ab_trials: int = 50) -> CompileResult:
        """Profile, identify the hottest path, compile it, A/B test, hot-swap.

        If the hottest path doesn't achieve speedup > 1.0, falls back to
        the next-warmest path.
        """
        profile = self.profile_tick(ticks=ticks)
        ranked = sorted(profile.items(), key=lambda kv: kv[1], reverse=True)
        log.info("Profile (ms/tick): %s", profile)

        dispatch = {
            "forward": self.compile_einsum,
            "routing": self.compile_routing,
        }

        best_result: Optional[CompileResult] = None
        for phase, _ in ranked:
            compiler_fn = dispatch.get(phase)
            if compiler_fn is None:
                continue
            result = compiler_fn(ab_trials=ab_trials)
            if result.validated and result.speedup > 1.0:
                result.profile_ms = profile
                return result
            if best_result is None or (result.validated and result.speedup > best_result.speedup):
                best_result = result

        if best_result is not None:
            best_result.profile_ms = profile
            return best_result

        return CompileResult(
            target="none", original=self._noop, compiled=self._noop,
            speedup=1.0, validated=False, hot_swapped=False,
            error="No path achieved speedup > 1.0",
            profile_ms=profile,
        )

    # ── Restore ─────────────────────────────────────────────

    def restore_original(self) -> int:
        """Rollback every hot-swap performed by this compiler instance.

        Returns the number of functions restored.
        """
        count = 0
        for key in list(self._originals.keys()):
            if self._restore(key):
                count += 1
                log.info("↩️  Restored %s", key)
        self.results.clear()
        return count

    # ── Helpers ─────────────────────────────────────────────

    @staticmethod
    def _noop(*args, **kwargs):
        """No-op placeholder for unavailable backends."""
        pass

    def report(self) -> str:
        """Human-readable report of compiled paths."""
        lines = ["=== RoomGridCompiler Report ==="]
        if not self.results:
            lines.append("Nothing compiled yet.")
            return "\n".join(lines)
        for target, r in self.results.items():
            status = "✅" if r.hot_swapped else "⚠️"
            if not r.validated:
                status = "❌"
            lines.append(
                f"  {status} {target:<10} speedup={r.speedup:>5.2f}×  "
                f"validated={r.validated}  swapped={r.hot_swapped}"
            )
            if r.error:
                lines.append(f"     error: {r.error}")
        return "\n".join(lines)
