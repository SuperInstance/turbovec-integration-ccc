# Contributing to turbovec-integration-ccc

## Getting Started

```bash
git clone https://github.com/SuperInstance/turbovec-integration-ccc.git
cd turbovec-integration-ccc
pip install -e ".[dev]"
```

## Workflow

1. Fork or branch from `main`
2. Make changes
3. `make lint && make test`
4. Open PR

## Modules

- `compiler/` — FLUX compiler integration
- `ethos/` — Agent ethos (character/tone) management
- `grammar/` — Grammar parsing
- `nerve/` — Routing and tick dispatch
- `nexus/` — Federation and mesh
- `swarm/` — Agent swarm coordination

## Release

Tag `v*` triggers automated PyPI publish.

## License

MIT — Fleet knowledge belongs to the fleet.
