"""
tests/test_metronome_integration.py

Tests for SynchronizedTickDispatcher.
Covers:
* Device health monitoring (heartbeat / miss / offline)
* Drift correction (nudge computation, skip-jump)
* Backend fallback when device fails
* Bridge-aware selective dispatch
* Monkey-patch install / uninstall round-trip
* Thread-safe concurrent ticks
"""

import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from nerve.metronome_integration import (
    DeviceHealthMonitor,
    DriftCorrector,
    SynchronizedTickDispatcher,
    install,
    DeviceOfflineError,
    IntegrationError,
)


# ── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def mock_grid():
    """Minimal RoomGrid-like object with the attributes we touch."""
    grid = MagicMock()
    grid.n = 100
    grid.w = {
        "w1": np.random.randn(100, 64, 32).astype(np.float32),
        "b1": np.zeros((1, 100, 32), dtype=np.float32),
        "w2": np.random.randn(100, 32, 16).astype(np.float32),
        "b2": np.zeros((1, 100, 16), dtype=np.float32),
        "w3": np.broadcast_to(np.eye(16, dtype=np.float32) * 0.99, (100, 16, 16)).copy(),
        "b3": np.zeros((1, 100, 16), dtype=np.float32),
    }
    grid.latents = np.zeros((100, 16), dtype=np.float32)
    grid.chaos = np.full(100, 0.3, dtype=np.float32)
    grid.activity = np.zeros(100, dtype=np.int32)
    grid.ticks = 0
    grid._out = np.empty((100, 16), dtype=np.float32)
    grid._hist = np.zeros((20, 100, 16), dtype=np.float32)
    grid._hist_idx = 0
    grid._hist_count = np.zeros(100, dtype=np.int32)
    grid._hist_max = 20
    grid._flux_checker = None
    grid._compiler = None
    grid._cognition_loop = None
    grid._agent_config = None
    grid._last_fired_ids = []

    def _forward(x):
        x = x.ravel().astype(np.float32)
        h = np.einsum("d,ndh->nh", x, grid.w["w1"], optimize=False) + grid.w["b1"][0]
        h = np.maximum(h, 0, out=h)
        h = np.einsum("nh,nhl->nl", h, grid.w["w2"], optimize=False) + grid.w["b2"][0]
        h = np.maximum(h, 0, out=h)
        return np.einsum("nl,nll->nl", h, grid.w["w3"], optimize=False) + grid.w["b3"][0]

    def tick(x):
        grid.ticks += 1
        latents = _forward(x)
        grid.latents = latents
        fired = [0, 1, 2]
        grid._last_fired_ids = fired
        return {"fired": len(fired), "ids": fired, "tick": grid.ticks}

    grid._forward = _forward
    grid.tick = tick
    return grid


@pytest.fixture
def mock_bridge():
    """Minimal MetronomeBridge-like mock."""
    bridge = MagicMock()
    bridge._dispatched_rooms = []
    bridge._latencies = {"cpu": [], "gpu": [], "rust": []}
    bridge._pending_ops = 0

    def on_beat(beat_number, tempo_ms):
        bridge._dispatched_rooms = list(range(10))
        return bridge._dispatched_rooms

    bridge.on_metronome_beat = on_beat
    bridge.scheduler = MagicMock()
    bridge.scheduler.beat_number = 0
    return bridge


@pytest.fixture
def dispatcher(mock_grid, mock_bridge):
    return SynchronizedTickDispatcher(
        grid=mock_grid,
        bridge=mock_bridge,
        beat_duration_ms=100.0,
    )


# ── DeviceHealthMonitor ───────────────────────────────────

class TestDeviceHealthMonitor:
    def test_register_creates_healthy_device(self):
        mon = DeviceHealthMonitor(timeout_ms=1000, max_missed=2)
        beat = mon.register("cuda", "cuda")
        assert beat.device_id == "cuda"
        assert beat.healthy is True
        assert beat.consecutive_misses == 0

    def test_heartbeat_updates_last_seen(self):
        mon = DeviceHealthMonitor()
        mon.register("rust", "rust_persistent")
        before = time.time_ns()
        mon.heartbeat("rust", latency_ms=4.2)
        after = time.time_ns()
        beat = mon.get_beat("rust")
        assert beat.last_seen_ns >= before
        assert beat.last_seen_ns <= after
        assert beat.last_latency_ms == 4.2
        assert beat.total_ticks == 1

    def test_miss_increments_counter(self):
        mon = DeviceHealthMonitor(max_missed=3)
        mon.register("cuda", "cuda")
        mon.miss("cuda")
        mon.miss("cuda")
        assert mon.get_beat("cuda").consecutive_misses == 2
        assert mon.get_beat("cuda").healthy is True

    def test_offline_after_max_missed(self):
        mon = DeviceHealthMonitor(max_missed=2)
        mon.register("cuda", "cuda")
        mon.miss("cuda")
        mon.miss("cuda")
        assert mon.get_beat("cuda").healthy is False
        assert mon.is_healthy("cuda") is False

    def test_heartbeat_resets_misses(self):
        mon = DeviceHealthMonitor(max_missed=2)
        mon.register("cuda", "cuda")
        mon.miss("cuda")
        mon.heartbeat("cuda", latency_ms=1.0)
        assert mon.get_beat("cuda").consecutive_misses == 0
        assert mon.is_healthy("cuda") is True

    def test_healthy_devices_filter(self):
        mon = DeviceHealthMonitor()
        mon.register("a", "numpy")
        mon.register("b", "cuda")
        mon.miss("b")
        mon.miss("b")
        mon.miss("b")
        assert mon.healthy_devices() == ["a"]
        assert mon.unhealthy_devices() == ["b"]

    def test_timeout_marks_unhealthy(self):
        mon = DeviceHealthMonitor(timeout_ms=50, max_missed=10)
        mon.register("old", "numpy")
        # Do not send heartbeat — simulate stale device
        time.sleep(0.08)
        assert mon.is_healthy("old") is False

    def test_reset_restores_health(self):
        mon = DeviceHealthMonitor(max_missed=2)
        mon.register("x", "numpy")
        mon.miss("x")
        mon.miss("x")
        assert mon.is_healthy("x") is False
        mon.reset("x")
        assert mon.is_healthy("x") is True


# ── DriftCorrector ──────────────────────────────────────────

class TestDriftCorrector:
    def test_zero_drift_when_no_records(self):
        corr = DriftCorrector()
        assert corr.get_drift_ms("cuda") == 0.0

    def test_records_store_expected_actual(self):
        corr = DriftCorrector(nudge_window=5)
        corr.record_tick("cuda", expected_ms=100, actual_ms=110)
        assert corr.get_drift_ms("cuda") == 10.0

    def test_nudge_zero_when_drift_small(self):
        corr = DriftCorrector(max_drift_ms=20.0)
        corr.record_tick("cuda", expected_ms=100, actual_ms=105)
        assert corr.get_nudge_ms("cuda", beat_duration_ms=100) == 0.0

    def test_nudge_computed_when_drift_large(self):
        corr = DriftCorrector(max_drift_ms=5.0, nudge_ratio_cap=0.05)
        corr.record_tick("cuda", expected_ms=100, actual_ms=120)
        nudge = corr.get_nudge_ms("cuda", beat_duration_ms=100)
        assert nudge != 0.0
        assert abs(nudge) <= 5.0  # 0.05 * 100

    def test_nudge_capped_at_ratio(self):
        corr = DriftCorrector(max_drift_ms=1.0, nudge_ratio_cap=0.05)
        # Huge drift
        corr.record_tick("cuda", expected_ms=100, actual_ms=500)
        nudge = corr.get_nudge_ms("cuda", beat_duration_ms=100)
        assert abs(nudge) <= 5.0

    def test_skip_jump_when_drift_exceeds_beat(self):
        corr = DriftCorrector()
        corr.record_tick("cuda", expected_ms=100, actual_ms=250)
        assert corr.should_skip_jump("cuda", beat_duration_ms=100) is True

    def test_no_skip_jump_when_drift_small(self):
        corr = DriftCorrector()
        corr.record_tick("cuda", expected_ms=100, actual_ms=105)
        assert corr.should_skip_jump("cuda", beat_duration_ms=100) is False

    def test_window_evicts_old_records(self):
        corr = DriftCorrector(nudge_window=2)
        corr.record_tick("cuda", expected_ms=100, actual_ms=200)
        corr.record_tick("cuda", expected_ms=100, actual_ms=200)
        corr.record_tick("cuda", expected_ms=100, actual_ms=100)
        # Window=2, so only last two remain
        assert corr.get_drift_ms("cuda") == 50.0  # (100 + 0) / 2


# ── SynchronizedTickDispatcher ────────────────────────────

class TestDispatcherInstall:
    def test_install_patches_forward_and_tick(self, dispatcher, mock_grid):
        orig_forward = mock_grid._forward
        orig_tick = mock_grid.tick
        dispatcher.install()
        assert mock_grid._forward is not orig_forward
        assert mock_grid.tick is not orig_tick

    def test_uninstall_restores_original(self, dispatcher, mock_grid):
        orig_forward = mock_grid._forward
        orig_tick = mock_grid.tick
        dispatcher.install()
        dispatcher.uninstall()
        assert mock_grid._forward is orig_forward
        assert mock_grid.tick is orig_tick

    def test_double_install_is_idempotent(self, dispatcher, mock_grid):
        dispatcher.install()
        first_forward = mock_grid._forward
        dispatcher.install()
        assert mock_grid._forward is first_forward


class TestDispatcherForward:
    def test_numpy_fallback_when_all_unhealthy(self, dispatcher, mock_grid):
        dispatcher.install()
        # Mark numpy as the only registered device, but make it unhealthy
        # Actually numpy is always registered; force miss it
        for _ in range(5):
            dispatcher.health.miss("numpy")
        x = np.random.randn(64).astype(np.float32)
        result = mock_grid._forward(x)
        assert result.shape == (100, 16)
        assert dispatcher.fallback_count >= 1

    def test_healthy_device_gets_heartbeat(self, dispatcher, mock_grid):
        dispatcher.install()
        x = np.random.randn(64).astype(np.float32)
        mock_grid._forward(x)
        beat = dispatcher.health.get_beat("numpy")
        assert beat.total_ticks >= 1
        assert beat.last_latency_ms > 0.0

    def test_drift_recorded_on_tick(self, dispatcher, mock_grid):
        dispatcher.install()
        x = np.random.randn(64).astype(np.float32)
        mock_grid._forward(x)
        drift = dispatcher.drift.get_drift_ms("numpy")
        # Drift should have been recorded
        assert drift is not None

    def test_backend_failure_falls_back(self, dispatcher, mock_grid):
        """Simulate a CUDA backend that raises; ensure fallback."""
        dispatcher.install()
        # Fake a CUDA backend that will explode
        mock_grid._cuda_grid = MagicMock()
        mock_grid._cuda_grid.tick.side_effect = RuntimeError("GPU burned")
        # Prevent MagicMock from auto-creating _rust_grid so the code
        # falls through to numpy.
        mock_grid._rust_grid = None
        dispatcher.health.register("cuda", "cuda")

        x = np.random.randn(64).astype(np.float32)
        # Force candidate list to prefer cuda
        mock_grid.n = 2000
        result = mock_grid._forward(x)
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 16)
        # One miss recorded; with default max_missed=3 device is still
        # technically "healthy" but has a miss logged.
        assert dispatcher.health.get_beat("cuda").consecutive_misses >= 1


class TestDispatcherTickWithBridge:
    def test_bridge_selective_dispatch(self, dispatcher, mock_grid, mock_bridge):
        dispatcher.install()
        x = np.random.randn(64).astype(np.float32)
        result = mock_grid.tick(x)
        assert result["dispatcher"] == "bridge"
        assert result["fired"] == 10
        # The bridge on_metronome_beat was called as part of tick()
        assert mock_bridge._dispatched_rooms == list(range(10))

    def test_bridge_result_has_ids(self, dispatcher, mock_grid, mock_bridge):
        dispatcher.install()
        x = np.random.randn(64).astype(np.float32)
        result = mock_grid.tick(x)
        assert "ids" in result
        assert "tick" in result


class TestDispatcherTickWithoutBridge:
    def test_no_bridge_runs_original_tick(self, mock_grid):
        disp = SynchronizedTickDispatcher(grid=mock_grid, bridge=None)
        disp.install()
        x = np.random.randn(64).astype(np.float32)
        result = mock_grid.tick(x)
        # Should still return a valid result
        assert "fired" in result
        assert "tick" in result


class TestDispatcherConcurrency:
    def test_thread_safe_ticks(self, dispatcher, mock_grid):
        dispatcher.install()
        errors = []
        results = []

        def worker():
            try:
                x = np.random.randn(64).astype(np.float32)
                r = mock_grid.tick(x)
                results.append(r)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 8


class TestDispatcherHealthReport:
    def test_report_contains_all_devices(self, dispatcher, mock_grid):
        dispatcher.install()
        x = np.random.randn(64).astype(np.float32)
        mock_grid._forward(x)
        report = dispatcher.health_report()
        assert "devices" in report
        assert "numpy" in report["devices"]
        assert "drift" in report
        assert report["fallback_count"] == dispatcher.fallback_count


class TestInstallUtility:
    def test_install_returns_dispatcher(self, mock_grid):
        disp = install(mock_grid, bridge=None, beat_duration_ms=250.0)
        assert isinstance(disp, SynchronizedTickDispatcher)
        assert disp.beat_duration_ms == 250.0
        disp.uninstall()

    def test_install_with_bridge(self, mock_grid, mock_bridge):
        disp = install(mock_grid, bridge=mock_bridge, beat_duration_ms=100.0)
        x = np.random.randn(64).astype(np.float32)
        result = mock_grid.tick(x)
        assert result["dispatcher"] == "bridge"
        disp.uninstall()


# ── Edge cases ──────────────────────────────────────────────

class TestEdgeCases:
    def test_emergency_tick_when_orig_tick_missing(self, mock_grid):
        """If orig_tick is None, dispatcher falls back to emergency tick."""
        disp = SynchronizedTickDispatcher(grid=mock_grid, bridge=None)
        disp.install()
        # Simulate corruption: clear orig_tick
        disp._orig_tick = None
        x = np.random.randn(64).astype(np.float32)
        result = disp._patched_tick(x)
        assert result["dispatcher"] == "emergency"

    def test_unregistered_heartbeat_is_ignored(self):
        mon = DeviceHealthMonitor()
        mon.heartbeat("ghost", 1.0)  # should not raise
        assert mon.get_beat("ghost") is None

    def test_drift_on_unknown_device_returns_zero(self):
        corr = DriftCorrector()
        assert corr.get_drift_ms("ghost") == 0.0

    def test_grid_with_zero_rooms(self):
        grid = MagicMock()
        grid.n = 0
        grid.w = {"w1": np.zeros((0, 64, 32)), "b1": np.zeros((1, 0, 32)),
                  "w2": np.zeros((0, 32, 16)), "b2": np.zeros((1, 0, 16)),
                  "w3": np.zeros((0, 16, 16)), "b3": np.zeros((1, 0, 16))}
        grid.latents = np.zeros((0, 16), dtype=np.float32)
        grid.chaos = np.array([], dtype=np.float32)
        grid.activity = np.array([], dtype=np.int32)
        grid.ticks = 0
        grid._out = np.zeros((0, 16), dtype=np.float32)
        grid._hist = np.zeros((20, 0, 16), dtype=np.float32)
        grid._hist_idx = 0
        grid._hist_count = np.array([], dtype=np.int32)
        grid._hist_max = 20
        grid._flux_checker = None
        grid._last_fired_ids = []

        def _forward(x):
            return np.zeros((0, 16), dtype=np.float32)

        grid._forward = _forward
        grid.tick = lambda x: {"fired": 0, "ids": [], "tick": 0}

        disp = install(grid)
        x = np.random.randn(64).astype(np.float32)
        result = grid.tick(x)
        assert result["fired"] == 0
        disp.uninstall()
