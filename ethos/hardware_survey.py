"""Hardware survey stubs — minimal dataclasses for turbovec-integration-ccc."""

from __future__ import annotations

__all__ = [
    "CudaGPU",
    "CPUInfo",
    "MemoryInfo",
    "ThermalZone",
    "HardwareProfile",
]

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CudaGPU:
    index: int
    name: str
    total_memory_mb: float
    free_memory_mb: float
    compute_capability: str
    multiprocessor_count: int
    temperature_c: Optional[float] = None
    utilization_pct: Optional[float] = None
    power_draw_w: Optional[float] = None
    power_limit_w: Optional[float] = None


@dataclass
class CPUInfo:
    model: str
    cores_physical: int
    cores_logical: int
    frequency_mhz: Optional[float] = None
    l1_cache_kb: Optional[int] = None
    l2_cache_kb: Optional[int] = None
    l3_cache_kb: Optional[int] = None


@dataclass
class MemoryInfo:
    total_ram_mb: float
    available_ram_mb: float
    total_swap_mb: float
    available_swap_mb: float


@dataclass
class ThermalZone:
    name: str
    type: str
    temperature_c: Optional[float] = None


@dataclass
class HardwareProfile:
    hostname: str
    platform: str
    cpu: CPUInfo
    memory: MemoryInfo
    cuda_gpus: List[CudaGPU] = field(default_factory=list)
    igpu_available: bool = False
    igpu_name: Optional[str] = None
    npu_available: bool = False
    npu_name: Optional[str] = None
    thermal_zones: List[ThermalZone] = field(default_factory=list)
    python_version: str = ""
    torch_available: bool = False
    torch_version: Optional[str] = None
    numpy_available: bool = False
    numpy_version: Optional[str] = None
    directml_available: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)
