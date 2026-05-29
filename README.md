# turbovec-integration-ccc

**TurboVec acceleration for the CCC ecosystem** — high-performance vector operations compiled to native code for the constraint-conservation-compilation pipeline.

## What This Gives You

- **Native vector ops** — SIMD-accelerated dot products, norms, and distances
- **CCC integration** — drop-in acceleration for conservation-spectral computations
- **Zero-copy** — operates directly on numpy arrays without serialization
- **Fallback** — pure Python fallback when native compilation is unavailable

## How It Fits

Performance layer for the conservation-spectral ecosystem. Accelerates the vector operations that `vector-novelty` and `conservation-tomography` rely on.

## License

MIT
