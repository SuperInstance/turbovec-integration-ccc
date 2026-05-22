"""Metronome Integration — Synchronized multi-device dispatch for RoomGrid.

Wires MetronomeBridge into RoomGrid.tick() so that every tick is
heartbeat-monitored, drift-corrected, and resilient to device failure.

Architecture
------------
* DeviceHealthMonitor  — tracks liveness of cpu/gpu/rust backends
* DriftCorrector       — nudges tick timing when a device runs slow
* SynchronizedTickDispatcher — intercepts grid._forward() and routes
  to the healthiest available backend, falling back to numpy on failure.
* install()            — monkey-patches a RoomGrid instance in one call.
"""

from __future__ import annotations

__all__ = [
    "DeviceHeartbeat",
    "DeviceHealthMonitor",
    "DriftCorrector",
    "SynchronizedTickDispatcher",
    "install",
    "IntegrationError",
]

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ── Exceptions ─────────────────────────────────────────────

class IntegrationError(Exception):
    """Raised when the integration layer cannot recover."""


class DeviceOfflineError(IntegrationError):
    """Raised when a backend device is marked offline after repeated misses."""


# ── Data structures ─────────────────────────────────────────

@dataclass
class DeviceHeartbeat:
    """Snapshot of a compute device's health."""

    device_id: str
    device_type: str  # "numpy", "rust_persistent", "rust_oneshot", "cuda"
    last_seen_ns: int = 0
    last_latency_ms: float = 0.0
    rtt_ms: float = 0.0
    consecutive_misses: int = 0
    total_ticks: int = 0
    healthy: bool = True


# ── Protocols ───────────────────────────────────────────────

class RoomGridLike(Protocol):
    """Duck-typed RoomGrid — only the attributes we touch."""

    n: int
    w: dict
    latents: np.ndarray
    chaos: np.ndarray
    activity: np.ndarray
    ticks: int
    _out: np.ndarray
    _hist: np.ndarray
    _hist_idx: int
    _hist_count: np.ndarray
    _hist_max: int

    def tick(self, x: np.ndarray) -> dict:
        ...


class MetronomeBridgeLike(Protocol):
    """Duck-typed MetronomeBridge."""

    grid: RoomGridLike
    scheduler: Any
    _dispatched_rooms: List[int]
    _latencies: Dict[str, List[float]]
    _pending_ops: int

    def on_metronome_beat(self, beat_number: int, tempo_ms: float) -> List[int]:
        ...

    def dispatch_room(self, room_id: int, device: str = "cpu") -> None:
        ...

    def sync_devices(self) -> None:
        ...

    def get_latency_report(self) -> dict:
        ...


# ── DeviceHealthMonitor ─────────────────────────────────────

class DeviceHealthMonitor:
    """Tracks heartbeat health for every compute backend.

    A device is *healthy* if it has responded within ``timeout_ms`` and
    has not exceeded ``max_missed`` consecutive misses.
    """

    def __init__(
        self,
        timeout_ms: float = 5_000.0,
        max_missed: int = 3,
    ):
        self.timeout_ms = timeout_ms
        self.max_missed = max_missed
        self._beats: Dict[str, DeviceHeartbeat] = {}
        self._lock = threading.Lock()

    def register(
        self,
        device_id: str,
        device_type: str,
    ) -> DeviceHeartbeat:
        """Register a new device for monitoring."""
        beat = DeviceHeartbeat(
            device_id=device_id,
            device_type=device_type,
            last_seen_ns=time.time_ns(),
            healthy=True,
        )
        with self._lock:
            self._beats[device_id] = beat
        log.info("Health monitor registered %s (%s)", device_id, device_type)
        return beat

    def heartbeat(
        self,
        device_id: str,
        latency_ms: float,
        *,
        rtt_ms: float = 0.0,
    ) -> None:
        """Record a successful tick from *device_id*."""
        with self._lock:
            beat = self._beats.get(device_id)
            if beat is None:
                log.warning("Heartbeat from unregistered device %s", device_id)
                return
            beat.last_seen_ns = time.time_ns()
            beat.last_latency_ms = latency_ms
            beat.rtt_ms = rtt_ms
            beat.consecutive_misses = 0
            beat.total_ticks += 1
            beat.healthy = True

    def miss(self, device_id: str) -> None:
        """Record a failed tick from *device_id*."""
        with self._lock:
            beat = self._beats.get(device_id)
            if beat is None:
                return
            beat.consecutive_misses += 1
            if beat.consecutive_misses >= self.max_missed:
                if beat.healthy:
                    log.warning(
                        "Device %s marked OFFLINE after %d missed beats",
                        device_id,
                        beat.consecutive_misses,
                    )
                beat.healthy = False

    def is_healthy(self, device_id: str) -> bool:
        """Return True if *device_id* is currently responsive."""
        with self._lock:
            beat = self._beats.get(device_id)
            if beat is None:
                return False
            if not beat.healthy:
                return False
            elapsed_ms = (time.time_ns() - beat.last_seen_ns) / 1_000_000.0
            return elapsed_ms < self.timeout_ms

    def healthy_devices(self) -> List[str]:
        """List all device_ids that are currently healthy."""
        with self._lock:
            return [did for did, b in self._beats.items() if b.healthy]

    def unhealthy_devices(self) -> List[str]:
        """List all device_ids that are currently offline."""
        with self._lock:
            return [did for did, b in self._beats.items() if not b.healthy]

    def get_beat(self, device_id: str) -> Optional[DeviceHeartbeat]:
        """Return the heartbeat record for *device_id*."""
        with self._lock:
            return self._beats.get(device_id)

    def reset(self, device_id: str) -> None:
        """Reset health state for *device_id* (e.g. after recovery)."""
        with self._lock:
            beat = self._beats.get(device_id)
            if beat is not None:
                beat.consecutive_misses = 0
                beat.healthy = True
                beat.last_seen_ns = time.time_ns()

    def all_devices(self) -> List[DeviceHeartbeat]:
        """Return a snapshot of every registered device."""
        with self._lock:
            return list(self._beats.values())


# ── DriftCorrector ──────────────────────────────────────────

class DriftCorrector:
    """Measures per-device tick latency and applies phase nudges.

    If a device is consistently slower than the metronome beat duration,
    the corrector computes a *nudge* (small time delta) that the scheduler
    can apply to keep the fleet in phase.
    """

    def __init__(
        self,
        max_drift_ms: float = 5.0,
        nudge_window: int = 5,
        nudge_ratio_cap: float = 0.05,
    ):
        self.max_drift_ms = max_drift_ms
        self.nudge_window = nudge_window
        self.nudge_ratio_cap = nudge_ratio_cap
        self._records: Dict[str, List[Tuple[float, float]]] = {}
        self._lock = threading.Lock()

    def record_tick(
        self,
        device_id: str,
        expected_ms: float,
        actual_ms: float,
    ) -> None:
        """Store (expected, actual) for *device_id*."""
        with self._lock:
            recs = self._records.setdefault(device_id, [])
            recs.append((expected_ms, actual_ms))
            if len(recs) > self.nudge_window:
                recs.pop(0)

    def get_drift_ms(self, device_id: str) -> float:
        """Mean (actual - expected) over the nudge window.

        Positive drift = device is slower than the beat.
        """
        with self._lock:
            recs = self._records.get(device_id, [])
            if not recs:
                return 0.0
            return float(np.mean([a - e for e, a in recs]))

    def get_nudge_ms(self, device_id: str, beat_duration_ms: float) -> float:
        """Compute a phase nudge for *device_id*.

        Returns 0.0 if drift is within ``max_drift_ms``.
        Otherwise returns a nudge ≤ ``nudge_ratio_cap * beat_duration_ms``.
        """
        drift = self.get_drift_ms(device_id)
        if abs(drift) <= self.max_drift_ms:
            return 0.0
        raw_nudge = drift * 0.5  # gentle correction
        cap = self.nudge_ratio_cap * beat_duration_ms
        nudge = max(-cap, min(cap, raw_nudge))
        log.info(
            "DriftCorrector nudge for %s: %.3f ms (drift %.3f ms, cap %.3f ms)",
            device_id,
            nudge,
            drift,
            cap,
        )
        return nudge

    def should_skip_jump(self, device_id: str, beat_duration_ms: float) -> bool:
        """True if drift exceeds one full beat — needs hard correction."""
        drift = abs(self.get_drift_ms(device_id))
        return drift > beat_duration_ms

    def reset(self, device_id: str) -> None:
        """Clear history for *device_id*."""
        with self._lock:
            self._records.pop(device_id, None)


# ── SynchronizedTickDispatcher ──────────────────────────────

class SynchronizedTickDispatcher:
    """Wraps RoomGrid.tick() with heartbeat-aware multi-device dispatch.

    Responsibilities
    ----------------
    1. Intercept ``grid._forward()`` and route to the healthiest backend.
    2. Record a heartbeat on every successful forward pass.
    3. On failure, mark the device missed and retry with the next-best
       backend (eventually falling back to numpy).
    4. Feed timing data into DriftCorrector so the metronome stays in phase.
    5. If a MetronomeBridge is attached, delegate selective dispatch to it
       rather than running a full-grid tick every beat.

    Usage
    -----
        dispatcher = SynchronizedTickDispatcher(grid, bridge=bridge)
        dispatcher.install()          # monkey-patches grid
        # grid.tick() now goes through dispatcher
        dispatcher.uninstall()        # restores original
    """

    def __init__(
        self,
        grid: RoomGridLike,
        bridge: Optional[MetronomeBridgeLike] = None,
        health_monitor: Optional[DeviceHealthMonitor] = None,
        drift_corrector: Optional[DriftCorrector] = None,
        beat_duration_ms: float = 500.0,
    ):
        self.grid = grid
        self.bridge = bridge
        self.health = health_monitor or DeviceHealthMonitor()
        self.drift = drift_corrector or DriftCorrector()
        self.beat_duration_ms = beat_duration_ms

        # Stash original methods for uninstall
        self._orig_forward: Optional[Callable] = None
        self._orig_tick: Optional[Callable] = None

        # Register backends that the grid *might* use
        self._register_grid_backends()

        # Thread safety
        self._lock = threading.Lock()
        self._fallback_count: int = 0
        self._last_result: Optional[dict] = None

    # ── installation ────────────────────────────────────────

    def install(self) -> None:
        """Monkey-patch grid so tick() and _forward() route through us."""
        if self._orig_forward is not None:
            return  # already installed

        self._orig_forward = getattr(self.grid, "_forward", None)
        self._orig_tick = getattr(self.grid, "tick", None)

        # Patch _forward first (it's the inner compute kernel)
        self.grid._forward = self._patched_forward  # type: ignore[method-assign]
        # Patch tick (outer orchestrator — we wrap bridge logic here)
        self.grid.tick = self._patched_tick  # type: ignore[method-assign]

        log.info(
            "SynchronizedTickDispatcher installed on RoomGrid(n=%d)",
            self.grid.n,
        )

    def uninstall(self) -> None:
        """Restore original tick() and _forward()."""
        if self._orig_forward is not None:
            self.grid._forward = self._orig_forward  # type: ignore[method-assign]
            self._orig_forward = None
        if self._orig_tick is not None:
            self.grid.tick = self._orig_tick  # type: ignore[method-assign]
            self._orig_tick = None
        log.info("SynchronizedTickDispatcher uninstalled")

    # ── backend registration ──────────────────────────────

    def _register_grid_backends(self) -> None:
        """Auto-register backends based on what's available in the grid."""
        # Always register numpy (the universal fallback)
        self.health.register("numpy", "numpy")

        # Rust persistent backend
        if hasattr(self.grid, "_rust_grid") or _rust_lib_available():
            self.health.register("rust_persistent", "rust_persistent")

        # CUDA backend
        if hasattr(self.grid, "_cuda_grid") or _cuda_lib_available():
            self.health.register("cuda", "cuda")

    # ── patched _forward ──────────────────────────────────

    def _patched_forward(self, x: np.ndarray) -> np.ndarray:
        """Heartbeat-aware backend dispatch.

        Tries backends in order of speed, skipping unhealthy ones.
        Falls back to numpy einsum if every fast backend is offline.
        """
        # Ordered preference: cuda → rust_persistent → rust_oneshot → numpy
        candidates = self._build_candidate_list()

        for device_id in candidates:
            if not self.health.is_healthy(device_id):
                log.debug("Skipping unhealthy backend %s", device_id)
                continue

            start = time.perf_counter()
            try:
                result = self._dispatch_to_backend(device_id, x)
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000
                log.warning(
                    "Backend %s failed after %.2f ms: %s",
                    device_id,
                    elapsed_ms,
                    exc,
                )
                self.health.miss(device_id)
                continue

            elapsed_ms = (time.perf_counter() - start) * 1000
            self.health.heartbeat(device_id, latency_ms=elapsed_ms)
            self.drift.record_tick(
                device_id,
                expected_ms=self.beat_duration_ms,
                actual_ms=elapsed_ms,
            )
            return result

        # All backends failed — fallback to numpy (always works)
        log.error("All backends failed — falling back to numpy einsum")
        self._fallback_count += 1
        start = time.perf_counter()
        result = _numpy_forward(self.grid.w, x)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.health.heartbeat("numpy", latency_ms=elapsed_ms)
        return result

    def _build_candidate_list(self) -> List[str]:
        """Return backend preference order for this grid size."""
        n = self.grid.n
        # Large grids: CUDA first, then Rust, then numpy
        if n >= 1000 and self.health.is_healthy("cuda"):
            return ["cuda", "rust_persistent", "numpy"]
        if n >= 500 and self.health.is_healthy("rust_persistent"):
            return ["rust_persistent", "numpy"]
        if n >= 50 and self.health.is_healthy("rust_oneshot"):
            return ["rust_oneshot", "numpy"]
        return ["numpy"]

    def _dispatch_to_backend(self, device_id: str, x: np.ndarray) -> np.ndarray:
        """Call the actual backend."""
        if device_id == "cuda":
            cuda_grid = getattr(self.grid, "_cuda_grid", None)
            if cuda_grid is None:
                from nerve.cuda_bridge import PersistentCUDAGrid
                self.grid._cuda_grid = PersistentCUDAGrid(self.grid.n, self.grid.w)  # type: ignore[attr-defined]
                cuda_grid = self.grid._cuda_grid
            return cuda_grid.tick(x)  # type: ignore[attr-defined]

        if device_id == "rust_persistent":
            rust_grid = getattr(self.grid, "_rust_grid", None)
            if rust_grid is None:
                from nerve.room_grid import PersistentRustGrid
                self.grid._rust_grid = PersistentRustGrid(self.grid.n, self.grid.w)  # type: ignore[attr-defined]
                rust_grid = self.grid._rust_grid
            return rust_grid.tick(x)  # type: ignore[attr-defined]

        if device_id == "rust_oneshot":
            from nerve.room_grid import forward_rust_oneshot
            return forward_rust_oneshot(self.grid.w, x, self.grid.n)

        # numpy fallback
        return _numpy_forward(self.grid.w, x)

    # ── patched tick ──────────────────────────────────────

    def _patched_tick(self, x: np.ndarray) -> dict:
        """Tick wrapper that adds bridge-aware selective dispatch.

        If a MetronomeBridge is attached, we let the bridge decide
        *which* rooms to tick (selective dispatch). Otherwise we fall
        back to the original full-grid tick.
        """
        with self._lock:
            if self.bridge is not None:
                # The bridge handles beat-aware selective dispatch
                beat_number = getattr(self.bridge.scheduler, "beat_number", 0)
                tempo_ms = self.beat_duration_ms
                dispatched = self.bridge.on_metronome_beat(beat_number, tempo_ms)
                # Reconstruct a minimal result dict
                self._last_result = {
                    "fired": len(dispatched),
                    "ids": dispatched[:10],
                    "tick": getattr(self.grid, "ticks", 0),
                    "dispatcher": "bridge",
                }
                return self._last_result

            # No bridge — run the original tick (which routes through
            # our patched _forward for backend health monitoring)
            if self._orig_tick is not None:
                result = self._orig_tick(x)
            else:
                # Emergency fallback if somehow orig_tick is None
                result = self._emergency_tick(x)
            self._last_result = result
            return result

    def _emergency_tick(self, x: np.ndarray) -> dict:
        """Last-resort tick when everything else is broken.

        Pure numpy, no bridge, no compiled backends.
        """
        self.grid.ticks += 1  # type: ignore[operator]
        latents = _numpy_forward(self.grid.w, x)
        self.grid.latents = latents  # type: ignore[attr-defined]
        # Minimal bookkeeping
        return {"fired": 0, "ids": [], "tick": self.grid.ticks, "dispatcher": "emergency"}

    # ── public inspection ───────────────────────────────────

    @property
    def fallback_count(self) -> int:
        return self._fallback_count

    def health_report(self) -> dict:
        """Snapshot of every device + drift stats."""
        report: dict = {
            "devices": {},
            "drift": {},
            "fallback_count": self._fallback_count,
            "beat_duration_ms": self.beat_duration_ms,
        }
        for beat in self.health.all_devices():
            report["devices"][beat.device_id] = {
                "healthy": beat.healthy,
                "consecutive_misses": beat.consecutive_misses,
                "total_ticks": beat.total_ticks,
                "last_latency_ms": round(beat.last_latency_ms, 3),
            }
            drift = self.drift.get_drift_ms(beat.device_id)
            report["drift"][beat.device_id] = round(drift, 3)
        return report

    def apply_drift_corrections(self, scheduler: Any) -> None:
        """Push nudges into a scheduler that supports ``nudge_phase()``.

        Call this from the metronome loop after every beat.
        """
        for beat in self.health.all_devices():
            if not beat.healthy:
                continue
            nudge = self.drift.get_nudge_ms(
                beat.device_id, self.beat_duration_ms
            )
            if nudge != 0.0 and hasattr(scheduler, "nudge_phase"):
                scheduler.nudge_phase(nudge)
            if self.drift.should_skip_jump(beat.device_id, self.beat_duration_ms):
                if hasattr(scheduler, "jump_to_beat"):
                    # Hard correction: snap scheduler to a new beat number
                    current = getattr(scheduler, "beat_number", 0)
                    scheduler.jump_to_beat(current + 1)


# ── Helpers ───────────────────────────────────────────────

def _numpy_forward(w: dict, x: np.ndarray) -> np.ndarray:
    """Pure-numpy einsum fallback (copied from room_grid.py)."""
    x = x.ravel().astype(np.float32)
    h = np.einsum("d,ndh->nh", x, w["w1"], optimize=False) + w["b1"][0]
    h = np.maximum(h, 0, out=h)
    h = np.einsum("nh,nhl->nl", h, w["w2"], optimize=False) + w["b2"][0]
    h = np.maximum(h, 0, out=h)
    return np.einsum("nl,nll->nl", h, w["w3"], optimize=False) + w["b3"][0]


def _rust_lib_available() -> bool:
    """Probe whether the Rust shared library is loadable."""
    try:
        from ctypes import CDLL
        from pathlib import Path
        so = next(Path(__file__).parent.glob("target/release/libjepa_kernel.so"))
        CDLL(str(so))
        return True
    except Exception:
        return False


def _cuda_lib_available() -> bool:
    """Probe whether CUDA runtime is available."""
    try:
        from ctypes import CDLL
        CDLL("libcudart.so")
        return True
    except Exception:
        return False


# ── Convenience installer ─────────────────────────────────

def install(
    grid: RoomGridLike,
    bridge: Optional[MetronomeBridgeLike] = None,
    beat_duration_ms: float = 500.0,
) -> SynchronizedTickDispatcher:
    """One-call installation of the synchronized dispatcher.

    Returns the dispatcher instance so the caller can inspect health
    or uninstall later.
    """
    dispatcher = SynchronizedTickDispatcher(
        grid=grid,
        bridge=bridge,
        beat_duration_ms=beat_duration_ms,
    )
    dispatcher.install()
    return dispatcher
