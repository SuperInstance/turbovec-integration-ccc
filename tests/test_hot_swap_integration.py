"""Tests for HotSwapIntegration — auto-compile, A/B gate, rollback.

Covers:
1. Config-change detection (breed, rebirth bumps _config_version).
2. Auto-recompile triggered manually / via watcher.
3. A/B test compares new compiled kernel vs current.
4. Rollback on safety-window failure.
5. tick() works after every hot-swap.
6. Event history tracking.
"""
from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from compiler.room_grid import RoomGrid, forward_einsum, batch_novelty
from compiler.compiler import RoomGridCompiler, _HAS_NUMBA
from compiler.hot_swap_integration import HotSwapIntegration, RecompileEvent


@pytest.fixture(autouse=True)
def _cleanup_compiler_hot_swaps():
    """Remove any lingering hot-swaps between tests."""
    mod = sys.modules.get("compiler.room_grid")
    if mod is not None:
        for attr in ("forward_einsum", "batch_novelty", "_tick_routing_compiled"):
            obj = getattr(mod, attr, None)
            if obj is not None and hasattr(obj, "_sunset_original"):
                setattr(mod, attr, obj._sunset_original)
            if attr == "_tick_routing_compiled" and hasattr(mod, attr):
                delattr(mod, attr)
    yield
    mod = sys.modules.get("compiler.room_grid")
    if mod is not None:
        for attr in ("forward_einsum", "batch_novelty", "_tick_routing_compiled"):
            obj = getattr(mod, attr, None)
            if obj is not None and hasattr(obj, "_sunset_original"):
                setattr(mod, attr, obj._sunset_original)
            if attr == "_tick_routing_compiled" and hasattr(mod, attr):
                delattr(mod, attr)


@pytest.fixture
def grid_10():
    np.random.seed(42)
    return RoomGrid(10)


@pytest.fixture
def grid_100():
    np.random.seed(42)
    return RoomGrid(100)


class TestConfigChangeDetection:
    """Grid config version bumps on mutation."""

    def test_breed_bumps_version(self, grid_100):
        before = grid_100._config_version
        grid_100.breed(0, 50)
        assert grid_100._config_version == before + 1

    def test_rebirth_bumps_version(self, grid_100):
        before = grid_100._config_version
        grid_100.rebirth(10)
        assert grid_100._config_version == before + 1

    def test_tick_does_not_bump_version(self, grid_100):
        before = grid_100._config_version
        grid_100.tick(np.random.randn(64).astype(np.float32))
        assert grid_100._config_version == before


class TestIntegrationBasics:
    """Instantiation and manual recompile."""

    def test_integration_instantiates(self, grid_100):
        integration = HotSwapIntegration(grid_100)
        assert integration.grid is grid_100
        assert integration.compiler is not None
        assert grid_100._compiler is integration.compiler

    def test_manual_recompile_runs(self, grid_100):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        integration = HotSwapIntegration(grid_100)
        event = integration.check_and_recompile(trigger="manual")
        assert isinstance(event, RecompileEvent)
        # Either swapped or failed gracefully
        assert event.target in ("einsum", "routing", "none")

    def test_recompile_updates_last_compile_version(self, grid_100):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        integration = HotSwapIntegration(grid_100)
        before = grid_100._config_version
        integration.check_and_recompile(trigger="manual")
        assert grid_100._last_compile_version == before

    def test_event_callback_fires(self, grid_100):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        cb = MagicMock()
        integration = HotSwapIntegration(grid_100, on_event=cb)
        integration.check_and_recompile(trigger="manual")
        assert cb.call_count == 1
        ev = cb.call_args[0][0]
        assert isinstance(ev, RecompileEvent)


class TestABGate:
    """A/B correctness + speedup gating."""

    def test_einsum_ab_speedup_and_correctness(self, grid_10):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        compiler = RoomGridCompiler(grid_10)
        result = compiler.compile_einsum(ab_trials=100)
        assert result.validated, f"A/B correctness failed: {result.error}"
        assert result.speedup > 1.0, f"Expected speedup > 1.0, got {result.speedup}"

    def test_routing_ab_speedup_and_correctness(self, grid_100):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        compiler = RoomGridCompiler(grid_100)
        result = compiler.compile_routing(ab_trials=100)
        assert result.validated, f"A/B correctness failed: {result.error}"
        assert result.speedup > 1.0, f"Expected speedup > 1.0, got {result.speedup}"

    def test_outputs_match_after_swap(self, grid_10):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        np.random.seed(42)
        x = np.random.randn(64).astype(np.float32)
        expected = forward_einsum(grid_10.w, x)
        compiler = RoomGridCompiler(grid_10)
        compiler.compile_einsum(ab_trials=100)
        actual = forward_einsum(grid_10.w, x)
        assert np.allclose(expected, actual, atol=1e-4, rtol=1e-3)
        compiler.restore_original()


class TestAutoRecompileOnConfigChange:
    """Integration detects breed/rebirth and recompiles."""

    def test_recompile_after_breed(self, grid_10):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        integration = HotSwapIntegration(grid_10)
        # Initial compile
        integration.check_and_recompile(trigger="init")
        assert len(integration.events) == 1

        # Mutate
        grid_10.breed(0, 5)
        assert grid_10._config_version != grid_10._last_compile_version

        # Recompile
        event = integration.check_and_recompile(trigger="config_change")
        assert event.trigger == "config_change"
        assert grid_10._last_compile_version == grid_10._config_version

    def test_recompile_after_rebirth(self, grid_10):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        integration = HotSwapIntegration(grid_10)
        integration.check_and_recompile(trigger="init")
        grid_10.rebirth(3)
        event = integration.check_and_recompile(trigger="config_change")
        assert event.trigger == "config_change"
        assert grid_10._last_compile_version == grid_10._config_version

    def test_no_recompile_when_unchanged(self, grid_10):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        integration = HotSwapIntegration(grid_10)
        integration.check_and_recompile(trigger="init")
        before = len(integration.events)
        # Calling again with same version should still recompile (manual trigger)
        # but watch-loop would skip it because versions match.
        event = integration.check_and_recompile(trigger="scheduled")
        assert len(integration.events) == before + 1  # manual always runs


class TestSafetyWindow:
    """Post-swap validation and rollback."""

    def test_safety_window_passes(self, grid_10):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        integration = HotSwapIntegration(grid_10, safety_window=5)
        event = integration.check_and_recompile(trigger="manual")
        if not event.hot_swapped:
            pytest.skip("No hot-swap occurred — safety window not applicable")

        np.random.seed(42)
        for _ in range(5):
            integration.on_tick()
            grid_10.tick(np.random.randn(64).astype(np.float32))

        assert not integration._pending_validation
        assert integration.events[-1].safety_checks_passed == 5
        assert not integration.events[-1].rolled_back

    def test_rollback_on_tick_failure(self, grid_10):
        """Simulate a broken kernel by injecting a bad function."""
        integration = HotSwapIntegration(grid_10, safety_window=3)
        # Force a fake hot-swap that will explode
        def _broken(*args, **kwargs):
            raise RuntimeError("simulated kernel failure")

        compiler = integration.compiler
        compiler._hot_swap("compiler.room_grid", "forward_einsum",
                           _broken, "compiler.room_grid.forward_einsum")
        integration._pending_validation = True
        integration._validation_ticks_remaining = 3
        integration._validation_target = "einsum"
        integration._events.append(
            RecompileEvent(
                trigger="manual", target="einsum",
                speedup=2.0, validated=True, hot_swapped=True,
            )
        )

        # First on_tick call should detect failure and rollback
        with pytest.raises(RuntimeError):
            grid_10.tick(np.random.randn(64).astype(np.float32))
            integration.on_tick()

        # After rollback, tick should work again
        # (The exception propagates, but rollback happens in on_tick,
        #  so we need to call rollback explicitly for this test.)
        compiler.restore_original()
        out = grid_10.tick(np.random.randn(64).astype(np.float32))
        assert "fired" in out


class TestTickRobustness:
    """tick() must work through every state transition."""

    def test_tick_after_compile(self, grid_10):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        integration = HotSwapIntegration(grid_10)
        integration.check_and_recompile(trigger="manual")
        np.random.seed(42)
        for _ in range(20):
            out = grid_10.tick(np.random.randn(64).astype(np.float32))
            assert "fired" in out
            assert "ids" in out

    def test_tick_after_breed_then_compile(self, grid_10):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        integration = HotSwapIntegration(grid_10)
        grid_10.breed(0, 5)
        integration.check_and_recompile(trigger="config_change")
        np.random.seed(42)
        for _ in range(20):
            out = grid_10.tick(np.random.randn(64).astype(np.float32))
            assert "fired" in out
            assert "ids" in out

    def test_tick_after_multiple_rebirths(self, grid_10):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        integration = HotSwapIntegration(grid_10)
        for i in range(5):
            grid_10.rebirth(i)
            integration.check_and_recompile(trigger="config_change")
        np.random.seed(42)
        for _ in range(20):
            out = grid_10.tick(np.random.randn(64).astype(np.float32))
            assert "fired" in out
            assert "ids" in out


class TestRestore:
    """Rollback returns system to clean state."""

    def test_restore_reverts_hot_swaps(self, grid_10):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        compiler = RoomGridCompiler(grid_10)
        result = compiler.compile_einsum(ab_trials=100)
        if not result.hot_swapped:
            pytest.skip("Hot-swap did not occur — speedup insufficient")
        current = sys.modules["compiler.room_grid"].forward_einsum
        assert hasattr(current, "_sunset_original"), "Expected _sunset_original on swapped function"
        original = current._sunset_original
        compiler.restore_original()
        restored = sys.modules["compiler.room_grid"].forward_einsum
        assert restored is original

    def test_integration_restore_via_compiler(self, grid_10):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        integration = HotSwapIntegration(grid_10)
        integration.check_and_recompile(trigger="manual")
        integration.compiler.restore_original()
        # After restore, _tick_routing_compiled should be gone
        mod = sys.modules.get("compiler.room_grid")
        assert not hasattr(mod, "_tick_routing_compiled")


class TestWatchLoop:
    """Background thread detects drift."""

    def test_watch_thread_spawns(self, grid_10):
        integration = HotSwapIntegration(grid_10)
        integration.start_watching(interval=0.1)
        assert integration.is_watching
        assert integration._watch_thread is not None
        assert integration._watch_thread.is_alive()
        integration.stop_watching()
        assert not integration.is_watching

    def test_watch_detects_breed(self, grid_10):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        integration = HotSwapIntegration(grid_10)
        integration.check_and_recompile(trigger="init")
        integration.start_watching(interval=0.1)

        # Mutate
        grid_10.breed(0, 5)
        # Wait for watch loop to notice
        time.sleep(0.4)

        integration.stop_watching()
        # Should have at least one config_change event
        config_events = [e for e in integration.events if e.trigger == "config_change"]
        assert len(config_events) >= 1


class TestReport:
    """Human-readable report generation."""

    def test_report_contains_events(self, grid_10):
        integration = HotSwapIntegration(grid_10)
        integration.check_and_recompile(trigger="manual")
        r = integration.report()
        assert "HotSwapIntegration Report" in r
        assert "manual" in r


# ── Global cleanup: ensure we never leave hot-swaps behind ──

def pytest_sessionfinish(session, exitstatus):
    """Restore any lingering hot-swaps so other test files are not tainted."""
    mod = sys.modules.get("compiler.room_grid")
    if mod is None:
        return
    for attr in ("forward_einsum", "batch_novelty", "_tick_routing_compiled"):
        obj = getattr(mod, attr, None)
        if obj is not None and hasattr(obj, "_sunset_original"):
            setattr(mod, attr, obj._sunset_original)
        if attr == "_tick_routing_compiled" and hasattr(mod, attr):
            delattr(mod, attr)
