"""Thermal budget manager — minimal stub for turbovec-integration-ccc."""

from __future__ import annotations

__all__ = [
    "DeviceBudget",
    "DeviceType",
    "ThermalBudget",
    "DEFAULT_BUDGETS",
]

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DeviceType(Enum):
    GPU = "gpu"
    CPU = "cpu"
    IGPU = "igpu"
    NPU = "npu"


DEFAULT_BUDGETS: dict[DeviceType, int] = {
    DeviceType.GPU: 9,
    DeviceType.CPU: 36,
    DeviceType.IGPU: 14,
    DeviceType.NPU: 6,
}


@dataclass
class DeviceBudget:
    device_type: DeviceType
    max_agents: int
    current_agents: int = 0

    @property
    def available(self) -> int:
        return max(0, self.max_agents - self.current_agents)

    @property
    def utilization(self) -> float:
        if self.max_agents == 0:
            return 0.0
        return self.current_agents / self.max_agents

    def __repr__(self) -> str:
        return (
            f"DeviceBudget({self.device_type.value}, "
            f"{self.current_agents}/{self.max_agents}, "
            f"util={self.utilization:.0%})"
        )


class ThermalBudget:
    def __init__(
        self,
        budgets: dict[DeviceType, int] | None = None,
    ) -> None:
        config = budgets if budgets is not None else DEFAULT_BUDGETS
        self._devices: dict[DeviceType, DeviceBudget] = {
            dt: DeviceBudget(device_type=dt, max_agents=max_agents)
            for dt, max_agents in config.items()
        }
        self._allocations: dict[str, DeviceType] = {}
        self._lock = threading.Lock()

    @property
    def total_max(self) -> int:
        return sum(d.max_agents for d in self._devices.values())

    @property
    def total_current(self) -> int:
        return sum(d.current_agents for d in self._devices.values())

    def device_budget(self, device: DeviceType) -> DeviceBudget:
        return self._devices[device]

    def can_spawn(self, device: DeviceType) -> bool:
        with self._lock:
            db = self._devices.get(device)
            if db is None:
                return False
            return db.current_agents < db.max_agents

    def allocate(self, agent_id: str, device: DeviceType) -> bool:
        with self._lock:
            if agent_id in self._allocations:
                raise ValueError(
                    f"Agent {agent_id!r} already allocated to "
                    f"{self._allocations[agent_id].value}"
                )
            db = self._devices.get(device)
            if db is None or db.current_agents >= db.max_agents:
                return False
            db.current_agents += 1
            self._allocations[agent_id] = device
            return True

    def release(self, agent_id: str) -> bool:
        with self._lock:
            device = self._allocations.pop(agent_id, None)
            if device is None:
                return False
            self._devices[device].current_agents -= 1
            return True

    def get_device(self, agent_id: str) -> Optional[DeviceType]:
        with self._lock:
            return self._allocations.get(agent_id)

    def thermal_headroom(self) -> float:
        with self._lock:
            max_total = self.total_max
            if max_total == 0:
                return 0.0
            return self.total_current / max_total

    def reset(self) -> None:
        with self._lock:
            for db in self._devices.values():
                db.current_agents = 0
            self._allocations.clear()
