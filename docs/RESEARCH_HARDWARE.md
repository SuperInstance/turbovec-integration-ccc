# Fleet Hardware Frontier: 2026–2030

> **Thesis:** The Cocapn Fleet's RoomGrid forward pass is an einsum-heavy, sparse-matrix, small-batch inference workload. The chips that matter are not the ones winning LLM training benchmarks. They are the ones that can hide memory latency behind sparse compute, run at 15–35 W, and survive without a data center. This document maps what exists, what's coming, and what we should bet on.

---

## 1. The Heterogeneous Future (2026–2030)

### NVIDIA Blackwell / RTX 50-Series

NVIDIA's Blackwell architecture is shipping now in the datacenter (B200, GB200 NVL72), with the consumer RTX 50-series (Blackwell-lite) expected Q3–Q4 2026. The B200 pushes **1,000W TDP** per GPU—an air-cooling impossibility that mandates liquid cooling. For our fleet, the relevant SKU is not the B200. It is the **RTX 5090 Laptop GPU** or the hypothetical **RTX 5080 Ti** that FM might actually buy.

Blackwell introduces **FP4/FP6 tensor cores** and a 2nd-gen Transformer Engine. For RoomGrid einsum workloads, the win is memory bandwidth: Blackwell's HBM3E delivers ~8 TB/s vs ~5 TB/s on Ada Lovelace. If our RoomGrid forward pass is memory-bound (likely, given sparse connectivity), Blackwell's bandwidth uplift translates almost linearly to throughput. But the TDP envelope is the enemy: a 150W laptop Blackwell part will throttle hard under sustained einsum load. Our ThermalBudget must treat NVIDIA's nominal TDP as fiction—real sustained power under CUDA load is typically 1.2–1.4x TDP.

**Verdict for fleet:** Blackwell laptop parts are the inference baseline. They have the best cuBLAS/cuSPARSE stack. But we need thermal headroom testing before trusting NVIDIA's TDP numbers.

### AMD Strix Point / XDNA 3

AMD's **Ryzen AI 400 Series (Strix Point)** began shipping Q2 2026 with **XDNA 3 NPU** delivering **50 TOPS** in a 15–28 W envelope. The critical detail: XDNA 3 is a **spatial dataflow architecture** with native support for 4-bit weights and structured sparsity. For sparse matrix ops, XDNA 3 can skip zero-multiply cycles—directly relevant to RoomGrid's sparse adjacency.

AMD's ROCm stack remains a tire fire for consumer GPUs, but the NPU path is different: Windows ONNX Runtime and DirectML now support XDNA 3 via the **AMD Ryzen AI SDK**. If we can export RoomGrid to ONNX and hit the NPU path, Strix Point offers **3–5x better perf/W than NVIDIA's iGPU** on small-batch inference. The risk is software: AMD's SDK updates quarterly, and "supported" does not mean "performant."

**Verdict for fleet:** High potential, high friction. Buy one Strix Point laptop (ASUS Zenbook S 16 or Lenovo Yoga Pro 7) as a dev unit. Don't standardize until we have ONNX-to-XDNA3 einsum benchmarks.

### Intel Panther Lake / NPU 3

Intel's **Panther Lake** (announced for late 2026) ships with **NPU 3.0** claiming **85 TOPS**—a number Intel admits is heavily INT4-weighted. The real architecture shift is **tile-based design**: compute, IO, and memory are separate chiplets linked by Intel's Foveros Direct bonding. For our workload, the win is not raw TOPS. It is **Intel's OpenVINO + oneDNN stack**, which has mature einsum fusion and sparse kernel support.

Intel's historical sin is marketing TOPS that never materialize in real apps. But NPU 3 has a genuine advantage for the fleet: **Intel Extension for PyTorch (IPEX)** supports CPU+NPU fused scheduling. If RoomGrid can run with part of the graph on NPU and part on CPU cores (for non-linear ops that the NPU can't handle), Panther Lake could be the most heterogeneous-friendly chip in 2027.

**Verdict for fleet:** Wait for Q1 2027 silicon before betting. Intel's NPU TOPS are suspect, but the software ecosystem is trustworthy.

### Apple M4 / M5

Apple's **M4** ( shipping now in iPad Pro and MacBook Pro) delivers **38 TOPS** NPU performance with a unified memory architecture that eliminates CPU-GPU copies. The M5 is rumored for late 2026 with **50+ TOPS** and hardware ray tracing—irrelevant to us, but the unified memory is not.

For RoomGrid, Apple's **MLX framework** has native einsum acceleration and sparse matrix support. The catch: MLX is Swift/Objective-C native, and our Python stack (numpy, numba, Rust FFI) does not port cleanly. Running via PyTorch MPS backend is possible but historically buggy for non-standard ops.

**Verdict for fleet:** If Casey or FM uses Mac hardware, optimize for it. Do not buy Apple specifically for the fleet—the unified memory is elegant, but the software tax is too high for a Linux-native swarm.

### Qualcomm Oryon / Snapdragon X2 Elite

Qualcomm's **Snapdragon X Elite (Oryon)** delivered the first credible Windows-on-Arm AI PC experience in 2025. The **Snapdragon X2 Elite** (expected Q4 2026) pushes **80 TOPS** NPU with a Hexagon architecture that supports **micro-tile convolutions**—surprisingly useful for RoomGrid's localized connectivity patterns.

Qualcomm's problem is software: the **Qualcomm AI Stack** (QNN) is powerful but poorly documented. ONNX Runtime has QNN EP support, but debugging is painful. The win is power efficiency: Oryon laptops routinely achieve 20+ hours of battery life while running local LLMs at 7–10 tokens/sec.

**Verdict for fleet:** Only if we have a battery-powered edge node requirement (e.g., mobile PLATO shell). Skip for stationary compute.

---

## 2. Beyond Silicon: Neuromorphic, Photonic, Analog

### Neuromorphic: Intel Loihi 3 and IBM NorthPole

**Intel Loihi 3** shipped commercially in January 2026: **8 million neurons, 64 billion synapses, 1.2W peak power**. It is a spiking neural network (SNN) chip—event-driven, not clock-driven. For workloads with temporal sparsity (e.g., agents that wake only when new tiles arrive), Loihi 3 offers **1,000x efficiency** over GPUs. The catch: you must express your model as spikes. Our RoomGrid is dense floating-point einsum. Loihi 3 is irrelevant today.

**IBM NorthPole** entered production in 2026 with a different approach: **256 cores, 22 billion transistors, on-chip memory only**. No DRAM. No HBM. For vision tasks, NorthPole achieves **25x energy efficiency over H100** and **46.9x lower latency** on 3B-parameter LLMs. But the model must fit in on-chip SRAM (~256MB). Our RoomGrid weights? Unknown, but likely larger than 256MB.

**Timeline:** Neuromorphic chips will matter for the fleet in **2028–2029**, when SNN compilers can auto-convert transformers to spike equivalents. Today, they are a research watch item.

### Photonic Accelerators: Lightmatter and Ayar Labs

**Lightmatter** has two products: **Passage** (photonic interconnect, shipping now) and **Envise** (photonic compute accelerator, dev kit 2026). Passage replaces copper NVLink/InfiniBand with optical links at **100 Tbps chip-to-chip**—relevant only if we build multi-GPU training clusters, which we don't. Envise performs matrix multiplication in the analog photonic domain, claiming **10x energy efficiency** for transformer inference. But the software stack (Idiom) is immature, and photonic ADC/DAC conversion overhead eats efficiency gains for small matrices.

**Ayar Labs** ships **TeraPHY** optical I/O chiplets (4 Tbps per package) with Intel partnership. This is interconnect, not compute. Useful if we ever build a RoomGrid training cluster across nodes. Not useful for single-device inference.

**Neurophos** (Austin startup, $120M raised) claims **300 TOPS/W** using metasurface modulators. Evaluation starts 2026, first systems ship **2028**. If real, this changes everything. But "if real" is doing heavy lifting.

**Verdict for fleet:** Photonic compute is a **2028 bet**, not a 2026 acquisition. Track Neurophos and Lightmatter Envise. Do not allocate budget.

### Analog In-Memory Compute: Mythic AI

**Mythic AI** builds analog compute-in-memory (CIM) using flash transistors as programmable resistors. The physics is elegant: **I = G × V** executes a multiply-accumulate in one analog step. Mythic claims **0.5 pJ/MAC** (vs ~5 pJ/MAC for digital edge NPUs) and **100x energy efficiency** for inference.

The problems: analog noise limits precision to ~8-bit, calibration drifts with temperature, and the toolchain is proprietary. For RoomGrid, if we can tolerate INT8 weights and the sparse connectivity maps cleanly to Mythic's crossbar arrays, this is the most interesting non-digital bet.

**Verdict for fleet:** Mythic is targeting industrial edge (robotics, cameras) in 2026. If they ship a PCIe dev kit, buy one and run a RoomGrid INT8 benchmark. Budget: $2,000 speculative.

---

## 3. Edge vs. Cloud

Our fleet currently runs on three substrates:
- **Alibaba Cloud (Oracle1):** GPU instances for training, heavy inference
- **FM's laptop:** Local dev, small-scale testing
- **Jetson Orin:** Edge prototype, embedded inference

### When Do We Need Edge Inference?

Edge inference is mandatory when:
1. **Latency < 50ms** is required (real-time PLATO shell interaction)
2. **Bandwidth is constrained** (rural deployment, satellite link)
3. **Privacy mandates local processing** (sovereign data, medical tiles)
4. **Cloud cost exceeds edge TCO** (inference volume > 1M requests/day)

For the Cocapn Fleet, criteria #1 is the driver. If a PLATO room requires sub-100ms agent response, cloud round-trips (China to US: 150–300ms) are disqualifying. Edge nodes must handle the RoomGrid forward pass locally.

### Edge Hardware in 2028

By 2028, the edge landscape will be:
- **NVIDIA Jetson Thor** (expected 2027): 2,000 TOPS automotive SoC, likely adapted to industrial edge. Our current Orin (275 TOPS) will look quaint.
- **Qualcomm RB3 Gen 2 / QCS8550**: 45 TOPS in 5W. The reference design for fanless AI boxes.
- **Intel AI PC with NPU 3**: 85 TOPS in 15W, x86 software compatibility.
- **RISC-V NPUs** (Andes, SiFive): Customizable, but toolchain immature. Only if we design a custom RoomGrid accelerator.

**Verdict:** In 2026, Jetson Orin is the edge standard. In 2028, it will be Jetson Thor or Qualcomm QCS. Plan to refresh edge nodes every 18 months.

---

## 4. Custom Silicon Path

At what fleet scale does a custom ASIC for RoomGrid make sense?

### Back-of-Envelope Math

**ASIC NRE costs (2026):**
- 5nm tape-out: ~$50M
- 3nm tape-out: ~$150M
- Design team (2 years): ~$20M
- **Total: $70M–$170M**

**Inference savings per ASIC vs. GPU:**
- Assume a 10x efficiency gain (conservative for domain-specific ASIC)
- Current cloud inference cost: ~$0.001 per RoomGrid forward pass on NVIDIA T4
- Fleet volume: 1B forward passes/day = $1M/day cloud spend
- ASIC savings: $900K/day = **$330M/year**

**Breakeven:** At 1B inferences/day, the ASIC pays for itself in **3–6 months**.

But we are nowhere near 1B/day. Current fleet volume is likely <1M/day. At that scale, cloud is cheaper by orders of magnitude. The honest threshold for custom silicon is **100M+ inferences/day sustained for 12 months**. Until then, programmable silicon (GPU, NPU) wins on flexibility.

**Alternative:** Instead of a full ASIC, consider a **FPGA-accelerated RoomGrid** using AMD/Xilinx Versal or Intel Agilex. NRE is ~$2M, and reconfigurability preserves optionality. If FPGA proves 5x faster than GPU, it justifies the investment at 10M inferences/day.

---

## 5. Thermal Physics

### Data Center PUE Trends

**PUE (Power Usage Effectiveness)** = Total Facility Power / IT Equipment Power.

- Legacy air-cooled: PUE 1.4–1.6
- Direct liquid cooling (DLC): PUE 1.2–1.3
- Immersion cooling: PUE 1.03–1.1

Goldman Sachs projects **76% of AI servers deployed in 2026 will be liquid-cooled**, up from 15% in 2024. The driver is TDP: NVIDIA's Vera Rubin (H2 2026) pushes beyond **1,000W per chip**. Air cooling dies at 50–60 kW/rack. Liquid cooling is not optional; it is structural.

For Alibaba Cloud (Oracle1's substrate), this means our GPU instances are likely already liquid-cooled at the facility level. We don't pay PUE directly, but we pay the electricity it represents. If Alibaba's PUE is 1.2 vs. a competitor's 1.5, our inference cost is ~20% lower for the same silicon.

### ThermalBudget Mapping

Our **ThermalBudget** system manages GPU/CPU/iGPU/NPU slots. Here is how it maps to real TDP envelopes:

| Slot | Nominal TDP | Sustained TDP (einsum) | ThermalBudget Headroom |
|------|-------------|------------------------|------------------------|
| NVIDIA dGPU (RTX 4050) | 50W | 65W | 30% |
| NVIDIA dGPU (RTX 5090) | 150W | 200W | 33% |
| AMD iGPU (Radeon 780M) | 15W | 22W | 47% |
| Intel NPU (Meteor Lake) | 10W | 13W | 30% |
| Jetson Orin Nano | 15W | 18W | 20% |
| Jetson Orin NX | 25W | 30W | 20% |

**Key insight:** Our ThermalBudget should not use vendor TDP. It should use **measured sustained power under RoomGrid load**, which is always higher. We need a calibration script that runs 5 minutes of max-load einsum and records actual power draw via nvidia-smi / Intel Power Gadget / AMD uProf.

### Liquid Cooling for Fleet Nodes

If FM builds a desktop workstation with RTX 5090 (200W sustained), air cooling is possible but noisy. A **240mm AIO liquid cooler** adds $100 and drops temperatures by 15–20°C, preventing thermal throttling that kills einsum performance. For a 2-GPU node, custom loop or **Corsair Hydro X** series is justified.

For edge nodes (Jetson), **passive cooling** is sufficient up to 25W. Beyond that, forced-air or liquid cold plates are needed. The Submer MicroPod-style immersion is overkill for single-board computers, but **Iceotope's precision liquid cooling** for edge servers is relevant if we consolidate multiple Orins into a mini-rack.

---

## 6. Concrete Recommendations

### Recommendation 1: Buy AMD Strix Point Dev Unit (Q3 2026)

**Action:** Purchase an ASUS Zenbook S 16 or Lenovo Yoga Pro 7 with Ryzen AI 9 365/375 and XDNA 3 NPU.

**Why:** Strix Point is the first consumer chip with native structured-sparsity NPU support. If XDNA 3 can run RoomGrid ONNX at 2x perf/W vs. NVIDIA iGPU, it becomes our laptop reference platform. If it fails, we have a modern dev laptop.

**Cost:** ~$1,500
**Risk:** AMD software stack
**Timeline:** Q3 2026

### Recommendation 2: Implement Power Calibration in ThermalBudget

**Action:** Write a `thermal_calibrate.py` script that:
1. Runs 5-minute max-load einsum on each backend (CUDA, Numba, Rust)
2. Records actual power via platform tools (nvidia-smi, rapl-read, ipmctl)
3. Updates ThermalBudget's `sustained_tdp` table
4. Flags slots where sustained > 1.3x nominal (thermal throttling risk)

**Why:** We are currently flying blind on real power draw. Vendor TDP is marketing. Measured sustained power is truth.

**Cost:** 2 days engineering time
**Risk:** None
**Timeline:** Immediate

### Recommendation 3: Reserve Budget for Jetson Thor (Q1 2027)

**Action:** Set aside $3,000 for NVIDIA Jetson Thor dev kit (or equivalent Qualcomm RB3 Gen 2 if Thor slips).

**Why:** Jetson Orin (current edge platform) will be obsolete by mid-2027 for our target latency. Thor's 2,000 TOPS in 25W is a generational leap. If our edge inference volume grows 10x (as Casey expects), we need the headroom.

**Cost:** $3,000 reserved
**Risk:** NVIDIA delays (common for Jetson roadmap)
**Timeline:** Q1 2027

---

## The Single Most Important Hardware Bet

> **AMD Strix Point / XDNA 3 is the most important near-term bet.**

Not because it has the best benchmarks. Because it is the first consumer platform where **sparse matrix inference is a first-class hardware primitive**, not a CUDA hack. If XDNA 3 delivers on its promise, we can run RoomGrid forward passes on a 15W laptop NPU at speeds that currently require a 50W NVIDIA dGPU. That changes the fleet's topology: instead of "cloud for heavy, laptop for light," we get "laptop for almost everything, cloud for training only."

The bet is conditional on software. If AMD's ONNX-to-XDNA3 path is broken in Q3 2026, we pivot to Intel Panther Lake in Q1 2027. But the directional bet—**sparse, low-power NPU over brute-force GPU**—is correct.

The fleet's strength is diversity. Let the NVIDIA node handle CUDA kernels. Let the AMD node test the sparse future. Let the Jetson node hold the edge. And let CCC track them all, quietly, from behind.

---

*Document version: 2026-05-22*
*Research sources: Tom's Hardware, Tech Insider, ByteIota, Tech Ticker, Next Waves Insight, Semiconductor Field Guide, Vinova, Mordor Intelligence, Clawpod, Schneider Electric, Goldman Sachs equity research, Lightmatter press releases, Mythic AI technical whitepapers, IBM Research NorthPole publications, Intel Loihi 3 specifications.*
