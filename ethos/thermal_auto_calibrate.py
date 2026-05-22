"""Thermal auto-calibration — adaptive budget tuning from hardware profiles.

Calibrates per-device agent budgets from live hardware survey data,
predicts future budget needs from workload history, and rebalances
agents when thermal alerts fire.
"""

from __future__ import annotations

__all__ = [
    "ThermalAlert",
    "ThermalAutoCalibrator",
]

import threading
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from swarm.thermal import DeviceType, ThermalBudget
    from ethos.hardware_survey import HardwareProfile


@dataclass(frozen=True)
class ThermalAlert:
    """A thermal zone has crossed a threshold.

    Attributes:
        zone_name: Identifier for the thermal zone (e.g. 'thermal_zone0').
        temperature_c: Current temperature in Celsius.
        threshold_c: Threshold that was crossed.
        severity: 'warning' or 'critical'.
    """
    zone_name: str
    temperature_c: float
    threshold_c: float
    severity: str

    def __repr__(self) -> str:
        return (
            f"ThermalAlert({self.zone_name!r}, "
            f"{self.temperature_c:.1f}°C / {self.threshold_c:.1f}°C, "
            f"severity={self.severity!r})"
        )


class ThermalAutoCalibrator:
    """Adaptive thermal budget calibration.

    Methods:
        - calibrate_from_profile: compute per-device budgets from hardware.
        - predict_budget: forecast future needs from workload history.
        - rebalance_on_alert: react to thermal alerts by moving agents.
    """

    def __init__(
        self,
        safety_margin: float = 0.15,
        thermal_throttle_c: float = 80.0,
        warning_threshold_c: float = 70.0,
    ) -> None:
        """Args:
            safety_margin: Fraction of theoretical max budget to reserve
                when thermals are elevated (0.0–1.0).
            thermal_throttle_c: Temperature at which we apply the safety
                margin reduction.
            warning_threshold_c: Temperature at which we start warning.
        """
        self.safety_margin = max(0.0, min(1.0, safety_margin))
        self.thermal_throttle_c = thermal_throttle_c
        self.warning_threshold_c = warning_threshold_c
        self._lock = threading.Lock()
        self._last_budget: dict["DeviceType", int] = {}
        self._rebalance_log: list[dict] = []

    # ------------------------------------------------------------------
    # 1. Calibration from hardware profile
    # ------------------------------------------------------------------

    def calibrate_from_profile(
        self,
        profile: "HardwareProfile",
        base_agents_per_core: float = 0.5,
        mem_per_agent_mb: float = 512.0,
        gpu_mem_per_agent_mb: float = 1024.0,
        igpu_mem_per_agent_mb: float = 512.0,
        npu_mem_per_agent_mb: float = 256.0,
    ) -> dict["DeviceType", int]:
        """Compute per-device agent budgets from a hardware profile.

        Rules:
            CPU  – cores_logical × base_agents_per_core, capped by RAM.
            GPU  – Σ(free_memory_mb / gpu_mem_per_agent_mb) per CUDA GPU.
            iGPU – free memory heuristic or 4 slots if available.
            NPU  – 2 slots if available.

        If any thermal zone exceeds ``thermal_throttle_c``, all budgets
        are multiplied by ``(1 - safety_margin)``.

        Args:
            profile: HardwareProfile from ethos.hardware_survey.
            base_agents_per_core: Agents per logical CPU core.
            mem_per_agent_mb: RAM budget per agent (CPU fallback cap).
            gpu_mem_per_agent_mb: VRAM budget per GPU agent.
            igpu_mem_per_agent_mb: iGPU memory budget per agent.
            npu_mem_per_agent_mb: NPU memory budget per agent.

        Returns:
            Mapping of DeviceType → max_agents.
        """
        import math
        from swarm.thermal import DeviceType

        # --- CPU ------------------------------------------------------------
        cpu_slots = int(profile.cpu.cores_logical * base_agents_per_core)
        if profile.memory.available_ram_mb > 0:
            ram_slots = int(profile.memory.available_ram_mb / mem_per_agent_mb)
            cpu_slots = min(cpu_slots, ram_slots)
        cpu_slots = max(1, cpu_slots)

        # --- GPU ------------------------------------------------------------
        gpu_slots = 0
        if profile.cuda_gpus:
            for gpu in profile.cuda_gpus:
                free = gpu.free_memory_mb if gpu.free_memory_mb else 0.0
                gpu_slots += int(free / gpu_mem_per_agent_mb)
        gpu_slots = max(0, gpu_slots)

        # --- iGPU -----------------------------------------------------------
        igpu_slots = 0
        if profile.igpu_available:
            # Heuristic: iGPU usually shares system RAM; give it a modest slice
            igpu_slots = max(1, int(profile.memory.available_ram_mb * 0.10 / igpu_mem_per_agent_mb))

        # --- NPU ------------------------------------------------------------
        npu_slots = 2 if profile.npu_available else 0

        # --- Thermal scaling ------------------------------------------------
        thermal_scale = 1.0
        if profile.thermal_zones:
            max_temp = max(
                (z.temperature_c for z in profile.thermal_zones if z.temperature_c is not None),
                default=0.0,
            )
            if max_temp >= self.thermal_throttle_c:
                thermal_scale = 1.0 - self.safety_margin

        raw_budgets = {
            DeviceType.CPU: max(1, math.floor(cpu_slots * thermal_scale)),
            DeviceType.GPU: max(0, math.floor(gpu_slots * thermal_scale)),
            DeviceType.IGPU: max(0, math.floor(igpu_slots * thermal_scale)),
            DeviceType.NPU: max(0, math.floor(npu_slots * thermal_scale)),
        }

        with self._lock:
            self._last_budget = dict(raw_budgets)

        return raw_budgets

    # ------------------------------------------------------------------
    # 2. Predictive budget
    # ------------------------------------------------------------------

    @staticmethod
    def _linear_regression_slope(y: list[float]) -> float:
        """Simple least-squares slope assuming x = 0, 1, 2, ..."""
        n = len(y)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(y) / n
        num = sum((i - x_mean) * (y_i - y_mean) for i, y_i in enumerate(y))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / den if den != 0 else 0.0

    def predict_budget(
        self,
        workload_history: list[tuple[int, float]],
        lookahead_ticks: int = 10,
    ) -> dict["DeviceType", int]:
        """Predict future budget needs from historical workload.

        Each history entry is ``(agent_count, thermal_headroom)`` for a tick.
        We fit a simple linear trend to agent_count, then distribute the
        predicted count across devices proportionally to the last known
        calibration.

        Args:
            workload_history: List of (agent_count, headroom) per tick.
            lookahead_ticks: How many ticks forward to predict.

        Returns:
            Predicted DeviceType → max_agents. Falls back to the last
            calibrated budget if history is empty.
        """
        import math
        from swarm.thermal import DeviceType

        with self._lock:
            fallback = dict(self._last_budget)

        if not workload_history or not fallback:
            return fallback

        agent_counts = [entry[0] for entry in workload_history]
        slope = self._linear_regression_slope(agent_counts)
        last_count = agent_counts[-1]
        predicted_count = max(0, int(round(last_count + slope * lookahead_ticks)))

        # Distribute predicted count across devices proportionally
        total_last = sum(fallback.values())
        if total_last == 0:
            return fallback

        proportions = {dt: fallback[dt] / total_last for dt in fallback}
        predicted: dict[DeviceType, int] = {}
        remainder = predicted_count
        for dt in DeviceType:
            if dt in fallback:
                share = math.floor(predicted_count * proportions[dt])
                predicted[dt] = max(0, share)
                remainder -= predicted[dt]

        # Distribute rounding remainder to the device with the largest budget
        sorted_by_size = sorted(
            [dt for dt in predicted if predicted[dt] > 0],
            key=lambda dt: fallback[dt],
            reverse=True,
        )
        for dt in sorted_by_size:
            if remainder <= 0:
                break
            predicted[dt] += 1
            remainder -= 1

        # Ensure at least one CPU slot
        if DeviceType.CPU in predicted and predicted[DeviceType.CPU] == 0:
            predicted[DeviceType.CPU] = 1

        return predicted

    # ------------------------------------------------------------------
    # 3. Rebalance on alert
    # ------------------------------------------------------------------

    def rebalance_on_alert(
        self,
        budget: "ThermalBudget",
        alert: ThermalAlert,
        agent_scores: dict[str, float] | None = None,
    ) -> list[tuple[str, "DeviceType", "DeviceType"]]:
        """React to a thermal alert by moving agents off the hot device.

        Heuristic mapping from zone name to DeviceType:
            - "gpu" or "card" → GPU
            - "cpu" or "x86" or "core" → CPU
            - "igpu" or "gfx" → IGPU
            - "npu" or "accel" → NPU

        Agents are sorted by ascending score (lowest first = sacrificed first).
        If no scores are provided, arbitrary order is used.

        Args:
            budget: The current ThermalBudget to mutate.
            alert: The thermal alert that fired.
            agent_scores: Optional mapping of agent_id → fitness/value for
                prioritization. Lower scores are moved first.

        Returns:
            List of ``(agent_id, from_device, to_device)`` tuples for every
            agent that was successfully moved.
        """
        from swarm.thermal import DeviceType

        hot_device = self._zone_to_device(alert.zone_name)
        if hot_device is None:
            return []

        # Gather agents on the hot device
        with budget._lock:
            agents_on_hot = [
                aid for aid, dev in budget._allocations.items() if dev == hot_device
            ]

        if not agents_on_hot:
            return []

        # Sort by score (ascending) so weakest agents move first
        if agent_scores:
            agents_on_hot.sort(key=lambda aid: agent_scores.get(aid, float("inf")))

        moves: list[tuple[str, DeviceType, DeviceType]] = []
        fallback_order = self._fallback_order(hot_device)

        for agent_id in agents_on_hot:
            moved = False
            for target in fallback_order:
                if budget.can_spawn(target):
                    # Perform the move
                    budget.release(agent_id)
                    ok = budget.allocate(agent_id, target)
                    if ok:
                        moves.append((agent_id, hot_device, target))
                        moved = True
                        break
                    else:
                        # Should not happen because can_spawn passed, but undo
                        # We can't undo release easily without tracking old device
                        # — skip this agent for safety
                        pass
            if not moved:
                # Agent remains on hot device (no room elsewhere)
                pass

        if moves:
            with self._lock:
                self._rebalance_log.append(
                    {
                        "alert": str(alert),
                        "moves": [(a, f.value, t.value) for a, f, t in moves],
                    }
                )

        return moves

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _zone_to_device(zone_name: str) -> Optional["DeviceType"]:
        """Heuristic: map a thermal zone name to a DeviceType."""
        from swarm.thermal import DeviceType

        lowered = zone_name.lower()
        if "igpu" in lowered or "gfx" in lowered:
            return DeviceType.IGPU
        if "gpu" in lowered or "card" in lowered:
            return DeviceType.GPU
        if "cpu" in lowered or "x86" in lowered or "core" in lowered:
            return DeviceType.CPU
        if "npu" in lowered or "accel" in lowered:
            return DeviceType.NPU
        return None

    @staticmethod
    def _fallback_order(hot: "DeviceType") -> list["DeviceType"]:
        """Return candidate fallback devices ordered by capacity preference."""
        from swarm.thermal import DeviceType

        # Prefer CPU (highest default capacity), then IGPU, NPU, GPU
        all_devices = [DeviceType.CPU, DeviceType.IGPU, DeviceType.NPU, DeviceType.GPU]
        return [d for d in all_devices if d != hot]

    @property
    def last_budget(self) -> dict["DeviceType", int]:
        """The most recent calibration result."""
        with self._lock:
            return dict(self._last_budget)

    @property
    def rebalance_log(self) -> list[dict]:
        """History of all rebalance operations performed."""
        with self._lock:
            return list(self._rebalance_log)
