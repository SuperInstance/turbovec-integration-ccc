"""Tests for ThermalAutoCalibrator.

Covers:
1. Calibration from hardware profiles (normal, thermal throttling, edge cases).
2. Predictive budget from workload history (trend, empty, proportional).
3. Rebalance on thermal alert (zone mapping, moves, fallbacks, no-room).
"""

from __future__ import annotations

import pytest

from ethos.hardware_survey import (
    CPUInfo,
    CudaGPU,
    HardwareProfile,
    MemoryInfo,
    ThermalZone,
)
from ethos.thermal_auto_calibrate import ThermalAlert, ThermalAutoCalibrator
from swarm.thermal import DeviceType, ThermalBudget


# ── fixtures ──────────────────────────────────────────────────

@pytest.fixture
def cold_profile() -> HardwareProfile:
    """A beefy machine with 8 logical cores, 32 GB RAM, 2 GPUs, cool thermals."""
    return HardwareProfile(
        hostname="test-host",
        platform="linux",
        cpu=CPUInfo(model="Test CPU", cores_physical=4, cores_logical=8, frequency_mhz=3000),
        memory=MemoryInfo(
            total_ram_mb=32768,
            available_ram_mb=16384,
            total_swap_mb=4096,
            available_swap_mb=4096,
        ),
        cuda_gpus=[
            CudaGPU(
                index=0,
                name="RTX 4090",
                total_memory_mb=24576,
                free_memory_mb=12288,
                compute_capability="8.9",
                multiprocessor_count=128,
            ),
            CudaGPU(
                index=1,
                name="RTX 4090",
                total_memory_mb=24576,
                free_memory_mb=8192,
                compute_capability="8.9",
                multiprocessor_count=128,
            ),
        ],
        igpu_available=False,
        npu_available=False,
        thermal_zones=[ThermalZone(name="thermal_zone0", type="x86_pkg_temp", temperature_c=55.0)],
    )


@pytest.fixture
def hot_profile(cold_profile: HardwareProfile) -> HardwareProfile:
    """Same machine but thermally stressed."""
    cold_profile.thermal_zones = [
        ThermalZone(name="thermal_zone0", type="x86_pkg_temp", temperature_c=85.0),
        ThermalZone(name="thermal_zone1", type="x86_pkg_temp", temperature_c=92.0),
    ]
    return cold_profile


@pytest.fixture
def calibrator() -> ThermalAutoCalibrator:
    return ThermalAutoCalibrator(safety_margin=0.15, thermal_throttle_c=80.0)


# ── calibrate_from_profile ────────────────────────────────────

class TestCalibrateFromProfile:
    def test_basic_calibration(self, calibrator: ThermalAutoCalibrator, cold_profile: HardwareProfile):
        budgets = calibrator.calibrate_from_profile(cold_profile)
        # CPU: 8 cores * 0.5 = 4, capped by 16384 / 512 = 32 → 4
        assert budgets[DeviceType.CPU] == 4
        # GPU: (12288 + 8192) / 1024 = 20
        assert budgets[DeviceType.GPU] == 20
        assert budgets[DeviceType.IGPU] == 0
        assert budgets[DeviceType.NPU] == 0
        assert calibrator.last_budget == budgets

    def test_thermal_throttling(self, calibrator: ThermalAutoCalibrator, hot_profile: HardwareProfile):
        budgets = calibrator.calibrate_from_profile(hot_profile)
        scale = 1.0 - 0.15  # safety_margin
        # CPU was 4, scaled → floor(4 * 0.85) = 3
        assert budgets[DeviceType.CPU] == 3
        # GPU was 20, scaled → floor(20 * 0.85) = 17
        assert budgets[DeviceType.GPU] == 17

    def test_no_gpus(self, calibrator: ThermalAutoCalibrator):
        profile = HardwareProfile(
            hostname="no-gpu",
            platform="linux",
            cpu=CPUInfo(model="ARM", cores_physical=2, cores_logical=4),
            memory=MemoryInfo(total_ram_mb=4096, available_ram_mb=2048, total_swap_mb=0, available_swap_mb=0),
            cuda_gpus=[],
        )
        budgets = calibrator.calibrate_from_profile(profile)
        assert budgets[DeviceType.GPU] == 0
        assert budgets[DeviceType.CPU] == 2  # 4 cores * 0.5 = 2, capped by 2048/512=4 → 2

    def test_igpu_and_npu(self, calibrator: ThermalAutoCalibrator):
        profile = HardwareProfile(
            hostname="igpu-npu",
            platform="linux",
            cpu=CPUInfo(model="Ryzen", cores_physical=6, cores_logical=12),
            memory=MemoryInfo(total_ram_mb=16384, available_ram_mb=8192, total_swap_mb=0, available_swap_mb=0),
            cuda_gpus=[],
            igpu_available=True,
            igpu_name="Radeon 780M",
            npu_available=True,
            npu_name="Ryzen AI",
        )
        budgets = calibrator.calibrate_from_profile(profile)
        assert budgets[DeviceType.IGPU] > 0
        assert budgets[DeviceType.NPU] == 2

    def test_zero_ram_capped_to_one_cpu(self, calibrator: ThermalAutoCalibrator):
        profile = HardwareProfile(
            hostname="starved",
            platform="linux",
            cpu=CPUInfo(model="Old", cores_physical=1, cores_logical=1),
            memory=MemoryInfo(total_ram_mb=256, available_ram_mb=0, total_swap_mb=0, available_swap_mb=0),
            cuda_gpus=[],
        )
        budgets = calibrator.calibrate_from_profile(profile)
        assert budgets[DeviceType.CPU] == 1


# ── predict_budget ──────────────────────────────────────────────

class TestPredictBudget:
    def test_empty_history_returns_last_budget(self, calibrator: ThermalAutoCalibrator, cold_profile: HardwareProfile):
        calibrator.calibrate_from_profile(cold_profile)
        predicted = calibrator.predict_budget([], lookahead_ticks=10)
        assert predicted == calibrator.last_budget

    def test_flat_history_unchanged(self, calibrator: ThermalAutoCalibrator, cold_profile: HardwareProfile):
        calibrator.calibrate_from_profile(cold_profile)
        history = [(10, 0.5)] * 5
        predicted = calibrator.predict_budget(history, lookahead_ticks=10)
        # Flat trend → same as current count, distributed proportionally
        total = sum(predicted.values())
        assert total == 10

    def test_rising_trend_increases_budget(self, calibrator: ThermalAutoCalibrator, cold_profile: HardwareProfile):
        calibrator.calibrate_from_profile(cold_profile)
        # Agent count rises: 5, 6, 7, 8, 9 → slope ~1.0
        history = [(5 + i, 0.5) for i in range(5)]
        predicted = calibrator.predict_budget(history, lookahead_ticks=5)
        total = sum(predicted.values())
        assert total > 9  # should be ~14 (9 + 1.0*5)

    def test_falling_trend_decreases_budget(self, calibrator: ThermalAutoCalibrator, cold_profile: HardwareProfile):
        calibrator.calibrate_from_profile(cold_profile)
        history = [(20 - i, 0.9) for i in range(5)]
        predicted = calibrator.predict_budget(history, lookahead_ticks=5)
        total = sum(predicted.values())
        assert total < 20
        assert total >= 0

    def test_predicted_cpu_at_least_one(self, calibrator: ThermalAutoCalibrator, cold_profile: HardwareProfile):
        calibrator.calibrate_from_profile(cold_profile)
        # Very low count should still keep 1 CPU slot
        history = [(0, 0.0)] * 3
        predicted = calibrator.predict_budget(history, lookahead_ticks=1)
        assert predicted[DeviceType.CPU] >= 1

    def test_proportional_distribution(self, calibrator: ThermalAutoCalibrator, cold_profile: HardwareProfile):
        calibrator.calibrate_from_profile(cold_profile)
        history = [(10, 0.5)] * 5
        predicted = calibrator.predict_budget(history, lookahead_ticks=10)
        last = calibrator.last_budget
        total_last = sum(last.values())
        # Proportions should be roughly preserved
        for dt in DeviceType:
            if last[dt] == 0:
                assert predicted[dt] == 0
            else:
                expected_ratio = last[dt] / total_last
                actual_ratio = predicted[dt] / sum(predicted.values())
                assert abs(actual_ratio - expected_ratio) < 0.10


# ── rebalance_on_alert ──────────────────────────────────────────

class TestRebalanceOnAlert:
    def test_moves_agents_off_hot_gpu(self):
        budget = ThermalBudget(budgets={DeviceType.GPU: 2, DeviceType.CPU: 4})
        budget.allocate("a1", DeviceType.GPU)
        budget.allocate("a2", DeviceType.GPU)
        budget.allocate("a3", DeviceType.CPU)

        calibrator = ThermalAutoCalibrator()
        alert = ThermalAlert(zone_name="gpu_card0", temperature_c=88.0, threshold_c=80.0, severity="critical")
        moves = calibrator.rebalance_on_alert(budget, alert)

        assert len(moves) == 2
        for agent_id, from_dev, to_dev in moves:
            assert from_dev == DeviceType.GPU
            assert to_dev == DeviceType.CPU
            assert budget.get_device(agent_id) == DeviceType.CPU

    def test_falls_back_to_next_device_if_cpu_full(self):
        budget = ThermalBudget(budgets={DeviceType.GPU: 2, DeviceType.CPU: 1, DeviceType.IGPU: 2})
        budget.allocate("a1", DeviceType.GPU)
        budget.allocate("a2", DeviceType.GPU)
        budget.allocate("a3", DeviceType.CPU)  # CPU full

        calibrator = ThermalAutoCalibrator()
        alert = ThermalAlert(zone_name="gpu_card0", temperature_c=88.0, threshold_c=80.0, severity="critical")
        moves = calibrator.rebalance_on_alert(budget, alert)

        # One agent goes to CPU (a1), the other to IGPU (a2) because CPU is full
        destinations = {t for _, _, t in moves}
        assert DeviceType.CPU in destinations or DeviceType.IGPU in destinations
        assert len(moves) <= 2

    def test_respects_agent_scores(self):
        budget = ThermalBudget(budgets={DeviceType.GPU: 2, DeviceType.CPU: 2})
        budget.allocate("weak", DeviceType.GPU)
        budget.allocate("strong", DeviceType.GPU)

        calibrator = ThermalAutoCalibrator()
        alert = ThermalAlert(zone_name="gpu", temperature_c=85.0, threshold_c=80.0, severity="warning")
        scores = {"weak": 0.2, "strong": 0.9}
        moves = calibrator.rebalance_on_alert(budget, alert, agent_scores=scores)

        # Only one agent should move (CPU has 2 slots, GPU has 2, one slot free on CPU)
        assert len(moves) >= 1
        # weak has lower score so it should move first
        moved_ids = [m[0] for m in moves]
        if len(moved_ids) == 1:
            assert moved_ids[0] == "weak"

    def test_no_agents_on_hot_device(self):
        budget = ThermalBudget(budgets={DeviceType.GPU: 2, DeviceType.CPU: 2})
        budget.allocate("a1", DeviceType.CPU)

        calibrator = ThermalAutoCalibrator()
        alert = ThermalAlert(zone_name="gpu_card0", temperature_c=88.0, threshold_c=80.0, severity="critical")
        moves = calibrator.rebalance_on_alert(budget, alert)
        assert moves == []

    def test_unknown_zone_returns_empty(self):
        budget = ThermalBudget(budgets={DeviceType.GPU: 2, DeviceType.CPU: 2})
        budget.allocate("a1", DeviceType.GPU)

        calibrator = ThermalAutoCalibrator()
        alert = ThermalAlert(zone_name="battery", temperature_c=45.0, threshold_c=40.0, severity="warning")
        moves = calibrator.rebalance_on_alert(budget, alert)
        assert moves == []

    def test_rebalance_log_records_moves(self):
        budget = ThermalBudget(budgets={DeviceType.GPU: 1, DeviceType.CPU: 1})
        budget.allocate("a1", DeviceType.GPU)

        calibrator = ThermalAutoCalibrator()
        alert = ThermalAlert(zone_name="gpu", temperature_c=85.0, threshold_c=80.0, severity="warning")
        calibrator.rebalance_on_alert(budget, alert)

        log = calibrator.rebalance_log
        assert len(log) == 1
        assert "moves" in log[0]
        assert len(log[0]["moves"]) == 1

    def test_no_room_anywhere_leaves_agents(self):
        budget = ThermalBudget(budgets={DeviceType.GPU: 1, DeviceType.CPU: 1})
        budget.allocate("a1", DeviceType.GPU)
        budget.allocate("a2", DeviceType.CPU)  # CPU full

        calibrator = ThermalAutoCalibrator()
        alert = ThermalAlert(zone_name="gpu", temperature_c=85.0, threshold_c=80.0, severity="warning")
        moves = calibrator.rebalance_on_alert(budget, alert)
        assert moves == []  # nowhere to go
        assert budget.get_device("a1") == DeviceType.GPU  # stays put

    def test_zone_to_device_heuristics(self):
        calibrator = ThermalAutoCalibrator()
        assert calibrator._zone_to_device("gpu_card0") == DeviceType.GPU
        assert calibrator._zone_to_device("x86_pkg_temp") == DeviceType.CPU
        assert calibrator._zone_to_device("igpu_gfx") == DeviceType.IGPU
        assert calibrator._zone_to_device("npu_accel0") == DeviceType.NPU
        assert calibrator._zone_to_device("battery") is None

    def test_fallback_order_excludes_hot(self):
        calibrator = ThermalAutoCalibrator()
        order = calibrator._fallback_order(DeviceType.GPU)
        assert DeviceType.GPU not in order
        assert len(order) == 3

    def test_cpu_alert_moves_to_gpu(self):
        budget = ThermalBudget(budgets={DeviceType.GPU: 2, DeviceType.CPU: 1})
        budget.allocate("a1", DeviceType.CPU)

        calibrator = ThermalAutoCalibrator()
        alert = ThermalAlert(zone_name="x86_pkg_temp", temperature_c=85.0, threshold_c=80.0, severity="warning")
        moves = calibrator.rebalance_on_alert(budget, alert)

        assert len(moves) == 1
        assert moves[0] == ("a1", DeviceType.CPU, DeviceType.GPU)
        assert budget.get_device("a1") == DeviceType.GPU
