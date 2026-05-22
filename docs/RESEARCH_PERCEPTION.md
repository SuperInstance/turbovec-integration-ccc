# Fleet Perception Frontier: From Text to World

*Research memo for the Cocapn Fleet — extending our agent swarm beyond pure computation into multi-modal perception of the real world.*

---

## 1. Multi-Modal Signal Ingestion: Turning the World into Tiles

Our current architecture is elegant but blind. RoomGrid processes latent vectors through einsum. NerveTopology routes "tiles" — text and structured data from ZeroClaw agents. PLATO rooms are rich in narrative but empty of photons, pressure waves, and thermal gradients. The question is not *whether* to add perception, but *how* to do it without collapsing the tile abstraction that makes our routing layer clean.

### Vision Pipeline

The most immediate modality is vision. Webcam feeds and screen capture produce raw frames at 30–60 Hz. Feeding uncompressed RGB tensors into our latent space would be catastrophic — a single 1920×1080 frame is 6.2MB. Instead, we follow the pattern established by CLIP-era visual encoders and modern VLA (Vision-Language-Action) models: a **frozen perception frontend** that compresses frames into embedding vectors, which then enter our tile router as first-class citizens.

The specific architecture we should adopt is a **two-tier embedding system**:

1. **Perception encoder**: A lightweight vision model (SigLIP, MobileViT, or a custom CNN distillation) runs on-device and outputs a 512-dimensional embedding per frame. This happens at 5–10 Hz, not 30 Hz — we do not need cinematic framerate for agent decision-making.
2. **Tile synthesizer**: The embedding is wrapped in a tile envelope with metadata (timestamp, sensor ID, confidence, bounding-box summaries from a lightweight YOLO-nano head). This tile is indistinguishable from a text tile at the NerveTopology layer.

For screen capture — critical for PLATO shell monitoring and web automation — we use the same pipeline but with a domain-specific encoder fine-tuned on UI element detection. The [OSWorld](https://github.com/xlang-ai/OSWorld) benchmark demonstrated that agents navigating GUIs benefit enormously from grounding visual perception in structured UI representations rather than raw pixels.

### Audio Pipeline

Audio is trickier because its semantics are temporal. A 44.1kHz waveform is useless as a tile; we need to collapse time into meaning. The standard approach in embodied AI (Habitat 3.0, SoundSpaces 2.0) is to run a streaming STT model (Whisper-tiny or a custom EdgeASR) in parallel with an **acoustic event detector** that classifies non-speech sounds (door slams, keyboard clicks, alert tones).

The tile format for audio should include:
- Transcribed text (if speech)
- Event class label (if non-speech)
- Direction-of-arrival estimate (if microphone array)
- Energy envelope (for "urgency" heuristics)

Crucially, audio tiles should be **delta-encoded**: only emitted when the acoustic scene changes, not on every 20ms frame. This prevents the router from drowning in silence.

### Sensor Streams

Temperature, IMU, LiDAR point clouds, motion PIR — these are already low-bandwidth. The ROS2 sensor_msgs conventions give us a proven schema. Each sensor produces a JSON tile at its natural frequency, tagged with `sensor_type` and `physical_unit`. The NerveTopology router already handles heterogeneous tile types; the fiber matching logic just needs to learn that `temperature` and `lidar_scan` are valid tile genres alongside `zc_digest` and `plato_observation`.

### The Tile Abstraction Holds

The key insight: we do not need to rebuild NerveTopology. We need to **expand the tile ontology**. A vision tile and a text tile both carry a latent vector, metadata, and a routing key. The einsum contraction in RoomGrid does not care whether the vector came from BERT or a ResNet. This is the same principle that made [DeepSet](https://arxiv.org/abs/1703.06114) architectures successful in multi-modal fusion — treat everything as a set of embeddings, let attention sort it out.

---

## 2. Embodied Agents: The Lifecycle FSM Meets Physics

Our agent lifecycle — EGG → COMPETE → SUNSET — was designed for computational agents competing for relevance in a latent economy. When the agent controls a physical body, the lifecycle maps naturally to robotic operational modes, but each phase acquires physical constraints.

### EGG: Calibration and Body Discovery

In the computational realm, EGG is model initialization, warm-up inference, and self-test. For an embodied agent, EGG becomes **hardware-in-the-loop calibration**:

- **Actuator discovery**: The agent queries its body bus (CAN, EtherCAT, or ROS2 control topics) to enumerate degrees of freedom. A drone discovers four rotors; a mobile base discovers wheel encoders and an IMU.
- **Sensor registration**: The agent publishes its sensor suite to the fleet registry — "I have a front-facing 720p camera, a 16-beam LiDAR, and a microphone array." This is analogous to a computational agent advertising its model capacity.
- **Safety envelope learning**: During EGG, the agent establishes hard limits — maximum joint torque, collision zones, emergency stop triggers. This is the physical equivalent of a sandbox. We draw directly on [Isaac Sim](https://developer.nvidia.com/isaac-sim)'s approach of running the first N control cycles in simulation before touching real hardware.

### COMPETE: Task Execution with Physics

The COMPETE phase is where the agent pursues its objective. In the physical world, this is closed-loop control, but with a critical difference: the agent's "relevance" is no longer just tile generation throughput. It is task completion quality, energy efficiency, and safety adherence.

The mapping:
- **Exploration** → Active perception: moving the body to reduce uncertainty about the environment (information-gain navigation, as in [Habitat](https://aihabitat.org/)'s embodied exploration benchmarks).
- **Exploitation** → Grasping, manipulation, navigation: using learned policies to achieve physical goals.
- **Constraint Engine** → Hard real-time safety checks: collision avoidance, tip-over prevention, joint limit enforcement. These run on a separate thread or micro-controller, not in the Python agent loop.

### SUNSET: Return to Dock and Data Harvest

SUNSET for a computational agent is context compression and graceful termination. For a physical agent, it is **return-to-dock and data offloading**:
- Navigation to charging station using a pre-learned topological map (the physical equivalent of "saving state").
- Upload of logged sensor data and experience replay buffers to fleet storage.
- Diagnostic self-check: "Did I damage anything? Are my motors overheating?" — the embodied equivalent of a healthcheck.
- If diagnostics pass, the agent enters a low-power sleep state. If not, it flags itself for maintenance and refuses new task assignments.

### The Safety Boundary

The most important addition to the lifecycle is a **physical emergency stop** that overrides the FSM. If a LiDAR detects an obstacle at <0.5m while the agent is moving, the motor controller cuts power immediately — no agent deliberation required. This is the [Asimov](https://en.wikipedia.org/wiki/Three_Laws_of_Robotics)-style hard constraint that must live below the agent layer, in the same way our Constraint Engine lives below the tile router.

---

## 3. Real-Time Constraints: Where Does the Boundary Lie?

Our current RoomGrid tick loop runs at ~14ms for 500 rooms. That is 71 Hz — respectable for a Python-based latent space simulator. But physical perception has deadlines that do not negotiate.

### The Hierarchy of Latency Requirements

| Task | Deadline | Current RoomGrid | Gap |
|------|----------|-----------------|-----|
| Motor current loop | 0.1–1 ms | 14,000 ms | **4 orders of magnitude** |
| Balance control (humanoid) | 1–5 ms | 14,000 ms | **3 orders of magnitude** |
| Obstacle avoidance (emergency) | 5–10 ms | 14,000 ms | **3 orders of magnitude** |
| Audio processing (STT) | 100–300 ms | 14 ms | **RoomGrid is faster** |
| Vision object detection | 30–100 ms | 14 ms | **RoomGrid is faster** |
| Fleet consensus (10k agents) | 500 ms–2 s | 14 ms | **Plausible with batching** |

The pattern is stark: **low-level motor control cannot live in the agent loop**. It needs a separate real-time substrate.

### The Split-Architecture Solution

We adopt the [ROS2](https://docs.ros.org/en/humble/index.html) model of **hard real-time on micro-controllers, soft real-time on the agent host**:

1. **RTOS layer** (FreeRTOS, Zephyr, or ROS2's rclc on STM32): runs at 1kHz, handles motor control, IMU fusion, emergency stops. No Python, no garbage collection, no context windows.
2. **Agent host** (Linux SBC, Jetson, or x86 edge): runs our RoomGrid/NerveTopology stack at 10–50 Hz, receives compressed sensor tiles from the RTOS layer, emits high-level action commands ("navigate to waypoint A", "grasp object class 'cup'").
3. **Fleet cloud**: runs batch consensus, model updates, and long-horizon planning. Latency here is seconds to minutes.

This is the same separation used by [Boston Dynamics' Spot](https://www.bostondynamics.com/products/spot) and [Figure AI's humanoid](https://www.figure.ai/) architecture: a real-time safety kernel below a deliberative AI layer.

### The 14ms Tick is Not the Problem

Our 14ms tick is fine for **cognitive loop** tasks: routing tiles, updating latent state, selecting the next room to visit. The error would be to try to run motor PID inside that same loop. Instead, the agent emits **goal tiles** ("desired end-effector pose: [x,y,z,qw,qx,qy,qz]") and the RTOS layer handles the trajectory.

---

## 4. Simulation-to-Reality Gap: From Numpy to Noise

We test in numpy. Reality is noisy, non-stationary, and occasionally adversarial (a human deliberately blocking the robot's camera). Bridging this gap is the central challenge of embodied AI.

### Why Simulators Lie

[Isaac Sim](https://developer.nvidia.com/isaac-sim) and [Habitat](https://aihabitat.org/) give us photorealistic rendering and accurate physics, but they still lie in predictable ways:

1. **Contact dynamics**: Simulated grippers exhibit sticky contacts and perfect friction models. Real grippers slip, bounce, and deform soft objects unpredictably.
2. **Sensor noise**: Simulated LiDAR returns clean point clouds. Real LiDAR sees dust, rain, and multi-path reflections.
3. **Latency**: Simulation is synchronous — the agent gets a perfect observation, computes, acts. Reality has asynchronous sensor streams, network jitter, and actuator lag.
4. **Distribution shift**: The simulation has a finite set of objects, textures, and lighting conditions. Reality is open-world.

### Our Sim2Real Strategy

We adopt a **three-phase curriculum**:

**Phase 1: Domain Randomization in Simulation**
Before deploying any policy, we train with heavy randomization: lighting, texture, object mass, friction coefficients, sensor noise models, actuator delay. The [OpenAI Rubik's Cube](https://openai.com/research/solving-rubiks-cube) manipulation project demonstrated that policies trained with sufficient randomization transfer with minimal fine-tuning.

**Phase 2: Digital Twin with Real Sensor Injection**
We run Isaac Sim in parallel with the real robot, feeding real sensor data into the simulation state estimator. The simulation provides predictive rollouts; the real sensors correct the state. This is the [MIT Cheetah](https://news.mit.edu/topic/robots-cheetah) approach: model-based control with real-time state correction.

**Phase 3: Online Adaptation with Fleet Memory**
When an agent encounters a novel physical situation ("this surface is slipperier than training"), it should not learn from scratch. It queries the fleet's collective memory: "Has any agent encountered high-slip surfaces? What policy adjustments worked?" This is the embodied equivalent of our current tile-sharing economy, but for control parameters rather than text tiles.

### The Role of PLATO as a Sim2Real Bridge

PLATO rooms can function as **abstract simulators**. A PLATO room describing a kitchen layout can be rendered by an agent into a spatial occupancy grid, which then guides real-world navigation. The text-to-geometry pipeline is lightweight and avoids the rendering cost of full photorealistic simulation. This is our secret weapon: we have a narrative engine that can generate structured world descriptions faster than any game engine can render them.

---

## 5. Fleet as Sensor Network: 10,000 Eyes and Ears

At fleet scale, perception becomes a distributed systems problem. Ten thousand agents generate terabytes of sensor data per day. Not all of it is useful. Most of it is redundant.

### Distributed Perception Architecture

We structure the fleet into **sensor swarms** — groups of agents with overlapping fields of view or sensing modalities, coordinated by a local "anchor" agent:

- **Spatial swarms**: Agents in the same physical area (a warehouse floor, a city block) share a local coordinate frame and fuse their observations into a collective occupancy map.
- **Modal swarms**: Agents with the same sensor type (all drone cameras, all temperature loggers) share calibration parameters and anomaly detection models.
- **Task swarms**: Agents working the same objective (search-and-rescue, inventory scanning) share intermediate representations rather than raw data.

The anchor agent is not special — it is simply the agent with the lowest network latency to the cloud at that moment. If it fails, another agent is elected. This is [Raft consensus](https://raft.github.io/) at the edge.

### Consensus on Observations

When two agents observe the same event (a door opening, a temperature spike), they may disagree. Was it a door opening or a shadow passing the camera? We need **Byzantine-fault-tolerant sensor fusion**:

- Each agent emits an observation tile with a confidence score.
- The local swarm runs a lightweight consensus protocol (weighted voting, where weights are derived from each agent's calibration history and recent accuracy).
- Outlier agents are not punished — they are **quarantined** for re-calibration. An agent that consistently disagrees with its swarm may have a faulty sensor, not a faulty algorithm.

This is directly inspired by [SwarmLab](https://swarmlab.org/)'s work on multi-robot belief sharing and by the Byzantine fault tolerance literature in distributed systems.

### Byzantine Sensor Faults

A sensor can fail in ways that are indistinguishable from correct operation: a frozen camera that keeps emitting the last frame, a LiDAR with a partially blocked beam that produces plausible but wrong point clouds. We detect these by **cross-modal inconsistency**: if the camera says "no obstacles" but the bumper sensor triggers, the camera is flagged. This cross-validation is cheap and catches a surprising fraction of sensor faults.

---

## 6. Concrete Recommendations: Three Tractable Perception Modules

### Module A: Vision Tile Encoder (Near-Term, 1–2 months)

Build a **lightweight vision-to-tile pipeline** using a pre-trained SigLIP or CLIP-Mobile model, running on CPU or Jetson GPU. The pipeline:
1. Captures a frame from webcam or screen.
2. Runs detection (YOLO-nano) + encoding (SigLIP) in parallel.
3. Emits a tile: `{type: "vision", embedding: [...], detections: [{class: "cup", bbox: [...], confidence: 0.94}], timestamp: ...}`
4. Ingests into NerveTopology as a first-class tile type.

**Hardware target**: Any agent host with a USB camera or VNC access. **Software target**: 5–10 Hz on a Raspberry Pi 4, 30 Hz on a Jetson Nano.

This is the **single most tractable perception addition**. It requires no new infrastructure — just a new tile producer that speaks the existing protocol.

### Module B: Sim2Real Calibration Gym (Mid-Term, 3–4 months)

Create a **PLATO room that simulates a physical robot calibration scenario**. Agents enter the room, receive a digital twin of a simple robot (2-wheeled mobile base + camera), and must navigate a maze with randomized physics parameters. The room tracks:
- Number of collisions (safety metric)
- Time to goal (efficiency metric)
- Policy transfer success rate when the same agent is deployed on a real TurtleBot3 or similar platform.

This gives us a **quantified sim2real gap metric** for the fleet, and a training ground for embodied policies before they touch hardware.

### Module C: Fleet Sensor Consensus Protocol (Mid-Term, 4–6 months)

Implement a **weighted voting consensus layer** in NerveTopology for duplicate observations. When ≥3 agents in a spatial swarm report observations of the same object/event within a time window, the consensus module:
1. Clusters observations by semantic similarity (embedding space distance).
2. Computes a weighted consensus embedding.
3. Flags outlier agents for re-calibration.
4. Emits a single "verified" tile upstream, suppressing duplicates.

This reduces upstream bandwidth by an estimated 60–80% in dense deployments, and improves observation accuracy by filtering noisy singleton reports.

---

## Closing

The fleet's current blindness is not a flaw — it is a deliberate simplification that let us build the routing and competition layers first. Now that those layers are stable, adding perception is a matter of **expanding the tile ontology** and **respecting the real-time boundary** between deliberation and control.

The most important principle: **do not let the beauty of the physical world break the elegance of the tile abstraction**. Vision, audio, and sensors all become tiles. The RoomGrid does not know or care what produced them. The NerveTopology router treats a LiDAR scan and a ZeroClaw digest with the same einsum logic. Perception is just another signal source in a universe of signals.

*Written for the Cocapn Fleet, 2026-05-22.*
