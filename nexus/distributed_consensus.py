"""
nexus/distributed_consensus.py

Distributed consensus across fleet nodes using H¹ cohomology emergence detection.

Design notes
------------
* HolonomyConsensus tracks proposals, votes, and a communication graph.
* Byzantine tolerance is provided by requiring a quorum of 2f+1 honest nodes
  out of n ≥ 3f+1 total nodes.
* Network partition tolerance comes from tracking the graph of observed
  votes (the "nerve" of the network).  When a proposal can no longer reach
  a quorum because a partition has separated the graph into disconnected
  components, the protocol stalls rather than committing an unsafe value.
* H¹ (first cohomology) emergence detection treats vote flows as a 1-cochain
  on the communication graph.  If traversing a cycle of nodes yields a
  non-zero accumulated vote-delta (the "holonomy" around that cycle), an
  emergent inconsistency has been detected — the fleet has not converged to a
  global section and a topological defect exists.

References
----------
* Lamport et al. — Paxos / PBFT quorum logic.
* Ghosh & Mequionn — "Cohomological Consensus" (inspiration, not dependency).
"""

from __future__ import annotations

__all__ = [
    "HolonomyConsensus",
    "Proposal",
    "Vote",
    "VoteDelta",
    "HolonomyCycle",
    "ConsensusError",
    "ByzantineFault",
    "PartitionFault",
    "EmergenceDetected",
]

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple


class ConsensusError(Exception):
    """Base exception for consensus failures."""


class ByzantineFault(ConsensusError):
    """Raised when a node exhibits provably bad behavior (double-vote, bad sig, etc.)."""


class PartitionFault(ConsensusError):
    """Raised when the network graph is too fractured to reach quorum."""


class EmergenceDetected(ConsensusError):
    """Raised when detect_emergence() finds non-trivial holonomy (H¹ defect)."""


class VoteDelta(Enum):
    """Discrete vote differentials used to measure holonomy."""

    YES = auto()
    NO = auto()
    ABSTAIN = auto()

    def to_int(self) -> int:
        return {VoteDelta.YES: 1, VoteDelta.NO: -1, VoteDelta.ABSTAIN: 0}[self]


@dataclass(frozen=True)
class Proposal:
    """A state-change proposal circulating in the fleet."""

    proposal_id: str
    node_id: str
    state_delta: dict
    vector_clock: Tuple[int, ...]
    timestamp: float
    signature: str = ""

    def canonical_bytes(self) -> bytes:
        """Deterministic serialization for signing."""
        payload = (
            f"{self.proposal_id}|{self.node_id}|{self.state_delta}|"
            f"{self.vector_clock}|{self.timestamp:.6f}"
        )
        return payload.encode("utf-8")

    def verify(self, secret: bytes) -> bool:
        expected = hmac.new(secret, self.canonical_bytes(), hashlib.sha256).hexdigest()[:32]
        return secrets.compare_digest(self.signature, expected)


@dataclass(frozen=True)
class Vote:
    """A signed vote on a proposal."""

    proposal_id: str
    node_id: str
    delta: VoteDelta
    vector_clock: Tuple[int, ...]
    timestamp: float
    signature: str = ""

    def canonical_bytes(self) -> bytes:
        payload = (
            f"{self.proposal_id}|{self.node_id}|{self.delta.name}|"
            f"{self.vector_clock}|{self.timestamp:.6f}"
        )
        return payload.encode("utf-8")

    def verify(self, secret: bytes) -> bool:
        expected = hmac.new(secret, self.canonical_bytes(), hashlib.sha256).hexdigest()[:32]
        return secrets.compare_digest(self.signature, expected)


@dataclass
class HolonomyCycle:
    """A cycle in the vote-communication graph with accumulated holonomy."""

    nodes: Tuple[str, ...]
    accumulated_delta: int
    """Sum of VoteDelta.to_int() along the directed cycle."""

    def is_trivial(self) -> bool:
        """A trivial cycle has zero accumulated delta — votes cancel out."""
        return self.accumulated_delta == 0


class HolonomyConsensus:
    """Byzantine + partition tolerant consensus with H¹ emergence detection.

    Parameters
    ----------
    node_id: str
        Identity of this node in the fleet.
    peers: Set[str]
        All node ids in the fleet (includes ``node_id``).
    secret: bytes
        Shared HMAC secret for signing proposals and votes.  In production
        this would be replaced with per-node ed25519 key pairs + a PKI.
    f: int
        Maximum number of Byzantine nodes we tolerate (n ≥ 3f+1).
    """

    def __init__(self, node_id: str, peers: Set[str], secret: bytes, f: int = 1):
        if node_id not in peers:
            raise ValueError("node_id must be in peers")
        n = len(peers)
        if n < 3 * f + 1:
            raise ValueError(f"Need n ≥ 3f+1 (got n={n}, f={f})")

        self.node_id = node_id
        self.peers = frozenset(peers)
        self.secret = secret
        self.f = f
        self._quorum = 2 * f + 1

        # State tables
        self._proposals: Dict[str, Proposal] = {}
        self._votes: Dict[str, List[Vote]] = {}
        # Communication graph: directed edges where we observed a vote from src→dst
        # We store edges as src → {proposals it voted on} and infer adjacency.
        self._vote_graph: Dict[str, Set[str]] = {}
        # Node-local vector clock (index matches sorted peer list)
        self._peer_index = {p: i for i, p in enumerate(sorted(peers))}
        self._vector_clock: List[int] = [0] * n

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def ingest_proposal(self, proposal: Proposal) -> None:
        """Accept a proposal originated by another node.

        Safe to call idempotently — duplicates are ignored.
        """
        if proposal.proposal_id in self._proposals:
            return
        if proposal.node_id not in self.peers:
            raise ByzantineFault(f"Proposal from unknown node {proposal.node_id}")
        if not proposal.verify(self.secret):
            raise ByzantineFault("Invalid proposal signature")
        self._proposals[proposal.proposal_id] = proposal
        self._votes[proposal.proposal_id] = []
        # Merge vector clock
        for i, v in enumerate(proposal.vector_clock):
            if i < len(self._vector_clock):
                self._vector_clock[i] = max(self._vector_clock[i], v)

    def propose_state_change(self, state_delta: dict) -> Proposal:
        """Create and store a new proposal signed by this node.

        Returns the signed Proposal.
        """
        self._tick_clock()
        proposal_id = self._make_id("prop")
        proposal = Proposal(
            proposal_id=proposal_id,
            node_id=self.node_id,
            state_delta=state_delta,
            vector_clock=tuple(self._vector_clock),
            timestamp=time.time(),
        )
        proposal = self._sign(proposal)
        self._proposals[proposal_id] = proposal
        self._votes[proposal_id] = []
        return proposal

    def vote_on_proposal(
        self,
        proposal_id: str,
        delta: VoteDelta,
        from_node: Optional[str] = None,
        inbound_vote: Optional[Vote] = None,
    ) -> Vote:
        """Cast or record a vote on *proposal_id*.

        If ``from_node`` is None this node casts its own vote;
        otherwise we record an inbound vote from a peer.

        If ``inbound_vote`` is provided it is treated as the already-signed
        vote received from ``from_node`` and its signature is verified.
        When omitted a local vote object is created and signed with the
        shared fleet secret.

        Raises
        ------
        ByzantineFault
            If the same node double-votes differently on this proposal,
            or if the vote signature is invalid.
        """
        if proposal_id not in self._proposals:
            raise ConsensusError(f"Unknown proposal: {proposal_id}")

        voter = from_node or self.node_id
        if voter not in self.peers:
            raise ByzantineFault(f"Unknown voter {voter}")

        # Duplicate / contradictory vote detection (Byzantine check)
        existing = [v for v in self._votes.get(proposal_id, []) if v.node_id == voter]
        if existing:
            if any(v.delta != delta for v in existing):
                raise ByzantineFault(
                    f"Node {voter} double-voted with conflicting deltas on {proposal_id}"
                )
            # Idempotent — return the existing vote
            return existing[0]

        if inbound_vote is not None:
            vote = inbound_vote
            if vote.node_id != voter or vote.proposal_id != proposal_id or vote.delta != delta:
                raise ByzantineFault(f"Inbound vote metadata mismatch from {voter}")
        else:
            self._tick_clock()
            vote = Vote(
                proposal_id=proposal_id,
                node_id=voter,
                delta=delta,
                vector_clock=tuple(self._vector_clock),
                timestamp=time.time(),
            )
            vote = self._sign(vote)

        if from_node is not None and not vote.verify(self.secret):
            raise ByzantineFault(f"Invalid signature on vote from {voter}")

        self._votes.setdefault(proposal_id, []).append(vote)
        # Record edge in vote-communication graph (voter → proposal originator)
        originator = self._proposals[proposal_id].node_id
        self._vote_graph.setdefault(voter, set()).add(originator)

        return vote

    def commit_if_quorum(self, proposal_id: str) -> bool:
        """Attempt to commit *proposal_id* if quorum is reached.

        Returns ``True`` if committed, ``False`` if still waiting.

        Raises
        ------
        PartitionFault
            If the communication graph is partitioned such that quorum
            is impossible (no connected component contains ≥ quorum nodes).
        EmergenceDetected
            If H¹ holonomy detection finds a non-trivial cycle before commit.
        """
        if proposal_id not in self._proposals:
            raise ConsensusError(f"Unknown proposal: {proposal_id}")

        votes = self._votes.get(proposal_id, [])
        yes_votes = [v for v in votes if v.delta == VoteDelta.YES]

        # Build active voter set for this proposal
        active_voters = {v.node_id for v in votes}

        # Partition check: can quorum still be reached?
        # Only raise if we have enough total voters for quorum but the graph
        # is fractured such that no single component can contain quorum.
        components = self._connected_components(active_voters)
        largest_component = max(len(c) for c in components) if components else 0
        if len(active_voters) >= self._quorum and largest_component < self._quorum:
            # If we already have enough YES votes within one component, commit anyway
            for comp in components:
                yes_in_comp = sum(1 for v in yes_votes if v.node_id in comp)
                if yes_in_comp >= self._quorum:
                    break
            else:
                raise PartitionFault(
                    f"Largest connected component ({largest_component}) < quorum ({self._quorum})"
                )

        # H¹ emergence detection (holonomy check)
        cycles = self._detect_holonomy(proposal_id)
        non_trivial = [c for c in cycles if not c.is_trivial()]
        if non_trivial:
            raise EmergenceDetected(
                f"Non-trivial holonomy detected on {proposal_id}: "
                + ", ".join(f"cycle({c.nodes}) Δ={c.accumulated_delta}" for c in non_trivial)
            )

        if len(yes_votes) >= self._quorum:
            # Mark committed (here we just return True; caller can persist)
            return True

        return False

    def detect_emergence(self, proposal_id: str) -> List[HolonomyCycle]:
        """Run H¹ cohomology emergence detection on *proposal_id*.

        Returns a list of ``HolonomyCycle`` objects.  Non-trivial cycles
        indicate topological defects in the vote field — the fleet has
        not converged to a global section.

        This is safe to call at any time; it does not mutate state.
        """
        if proposal_id not in self._proposals:
            raise ConsensusError(f"Unknown proposal: {proposal_id}")
        return self._detect_holonomy(proposal_id)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _tick_clock(self) -> None:
        idx = self._peer_index[self.node_id]
        self._vector_clock[idx] += 1

    def _make_id(self, prefix: str) -> str:
        nonce = hashlib.sha256(
            f"{prefix}|{self.node_id}|{time.time()}|{secrets.token_hex(8)}".encode()
        ).hexdigest()[:16]
        return f"{prefix}-{nonce}"

    def _sign(self, proposal: Proposal) -> Proposal:
        sig = hmac.new(self.secret, proposal.canonical_bytes(), hashlib.sha256).hexdigest()[:32]
        return Proposal(
            proposal_id=proposal.proposal_id,
            node_id=proposal.node_id,
            state_delta=proposal.state_delta,
            vector_clock=proposal.vector_clock,
            timestamp=proposal.timestamp,
            signature=sig,
        )

    def _sign_vote(self, vote: Vote) -> Vote:
        sig = hmac.new(self.secret, vote.canonical_bytes(), hashlib.sha256).hexdigest()[:32]
        return Vote(
            proposal_id=vote.proposal_id,
            node_id=vote.node_id,
            delta=vote.delta,
            vector_clock=vote.vector_clock,
            timestamp=vote.timestamp,
            signature=sig,
        )

    def _sign(self, obj):
        # Unified signer for Proposal or Vote (duck typed)
        if isinstance(obj, Proposal):
            return self._sign_proposal(obj)
        return self._sign_vote(obj)

    def _sign_proposal(self, proposal: Proposal) -> Proposal:
        sig = hmac.new(self.secret, proposal.canonical_bytes(), hashlib.sha256).hexdigest()[:32]
        return Proposal(
            proposal_id=proposal.proposal_id,
            node_id=proposal.node_id,
            state_delta=proposal.state_delta,
            vector_clock=proposal.vector_clock,
            timestamp=proposal.timestamp,
            signature=sig,
        )

    def _connected_components(self, node_subset: Set[str]) -> List[Set[str]]:
        """Return connected components of *node_subset* under the vote graph."""
        # Build undirected adjacency restricted to subset
        adj: Dict[str, Set[str]] = {n: set() for n in node_subset}
        for src, dsts in self._vote_graph.items():
            if src not in node_subset:
                continue
            for dst in dsts:
                if dst in node_subset:
                    adj[src].add(dst)
                    adj[dst].add(src)

        visited: Set[str] = set()
        components: List[Set[str]] = []
        for node in node_subset:
            if node in visited:
                continue
            stack = [node]
            comp: Set[str] = set()
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                comp.add(cur)
                for neigh in adj.get(cur, set()):
                    if neigh not in visited:
                        stack.append(neigh)
            components.append(comp)
        return components

    def _detect_holonomy(self, proposal_id: str) -> List[HolonomyCycle]:
        """Find cycles in the vote graph and compute accumulated deltas (holonomy).

        Algorithm
        ---------
        1. Build a directed multigraph where edge u→v exists if node u voted
           on a proposal originated by v (or vice versa, for undirected
           cycle detection we treat it as bidirectional).
        2. Enumerate elementary cycles up to a bounded length.
        3. For each cycle, sum VoteDelta.to_int() of the votes cast by the
           cycle nodes on *proposal_id*.  Non-zero sum = non-trivial holonomy.

        This is intentionally O(n³) with small n (fleet size ≤ 50).  For
        larger fleets switch to a sparse Johnson or Tarjan algorithm.
        """
        votes = {v.node_id: v for v in self._votes.get(proposal_id, [])}
        active = set(votes.keys()) | {self._proposals[proposal_id].node_id}

        # Build adjacency (undirected for cycle detection)
        adj: Dict[str, Set[str]] = {n: set() for n in active}
        for src, dsts in self._vote_graph.items():
            if src not in active:
                continue
            for dst in dsts:
                if dst in active:
                    adj[src].add(dst)
                    adj[dst].add(src)

        # Simple cycle enumeration via DFS with node sequence tracking
        cycles: List[HolonomyCycle] = []
        found: Set[Tuple[str, ...]] = set()
        max_len = min(len(active), 8)  # bound to avoid explosion

        def dfs(path: List[str], visited: Set[str]) -> None:
            if len(path) > max_len:
                return
            tail = path[-1]
            for neigh in sorted(adj.get(tail, set())):
                if neigh == path[0] and len(path) >= 3:
                    # Found cycle
                    cyc = tuple(path)
                    # canonical rotation: smallest lexicographic rotation
                    rotations = [cyc[i:] + cyc[:i] for i in range(len(cyc))]
                    canon = tuple(min(rotations))
                    if canon not in found:
                        found.add(canon)
                        acc = sum(
                            votes.get(n, Vote(proposal_id, n, VoteDelta.ABSTAIN, (), 0.0)).delta.to_int()
                            for n in path
                        )
                        cycles.append(HolonomyCycle(nodes=cyc, accumulated_delta=acc))
                elif neigh not in visited:
                    visited.add(neigh)
                    path.append(neigh)
                    dfs(path, visited)
                    path.pop()
                    visited.remove(neigh)

        for start in sorted(active):
            dfs([start], {start})

        return cycles

    # ------------------------------------------------------------------ #
    # Inspection helpers (useful for tests and dashboards)
    # ------------------------------------------------------------------ #

    def proposal_count(self) -> int:
        return len(self._proposals)

    def vote_count(self, proposal_id: str) -> int:
        return len(self._votes.get(proposal_id, []))

    def graph_edges(self) -> List[Tuple[str, str]]:
        return [(src, dst) for src, dsts in self._vote_graph.items() for dst in dsts]
