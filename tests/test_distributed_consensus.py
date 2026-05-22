"""
tests/test_distributed_consensus.py

Tests for HolonomyConsensus.
Covers happy-path consensus, Byzantine double-vote detection,
network partition handling, and H¹ holonomy emergence detection.
"""

import pytest
from nexus.distributed_consensus import (
    HolonomyConsensus,
    Proposal,
    Vote,
    VoteDelta,
    HolonomyCycle,
    ByzantineFault,
    PartitionFault,
    EmergenceDetected,
    ConsensusError,
)


@pytest.fixture
def secret():
    return b"fleet-shared-secret-001"


@pytest.fixture
def three_node(secret):
    peers = {"alpha", "beta", "gamma", "delta"}
    return {
        "alpha": HolonomyConsensus("alpha", peers, secret, f=1),
        "beta": HolonomyConsensus("beta", peers, secret, f=1),
        "gamma": HolonomyConsensus("gamma", peers, secret, f=1),
        "delta": HolonomyConsensus("delta", peers, secret, f=1),
    }


@pytest.fixture
def four_node(secret):
    peers = {"n1", "n2", "n3", "n4"}
    return {
        n: HolonomyConsensus(n, peers, secret, f=1)
        for n in peers
    }


# -------------------------------------------------------------------- #
# Basic construction
# -------------------------------------------------------------------- #


def test_init_requires_node_in_peers(secret):
    with pytest.raises(ValueError, match="node_id must be in peers"):
        HolonomyConsensus("x", {"a", "b"}, secret, f=1)


def test_init_enforces_3f_plus_1(secret):
    # n=3, f=1 is borderline (3*1+1=4 required)
    with pytest.raises(ValueError, match=r"n ≥ 3f\+1"):
        HolonomyConsensus("a", {"a", "b", "c"}, secret, f=1)
    # n=4, f=1 is okay
    HolonomyConsensus("a", {"a", "b", "c", "d"}, secret, f=1)


# -------------------------------------------------------------------- #
# propose_state_change
# -------------------------------------------------------------------- #


def test_propose_creates_signed_proposal(three_node):
    alpha = three_node["alpha"]
    p = alpha.propose_state_change({"temp": 42})
    assert isinstance(p, Proposal)
    assert p.node_id == "alpha"
    assert p.state_delta == {"temp": 42}
    assert p.verify(alpha.secret)
    assert alpha.proposal_count() == 1


def test_proposal_vector_clock_increments(three_node):
    alpha = three_node["alpha"]
    p1 = alpha.propose_state_change({"a": 1})
    p2 = alpha.propose_state_change({"b": 2})
    # alpha is index 0 in sorted peers (alpha, beta, gamma)
    assert p2.vector_clock[0] == p1.vector_clock[0] + 1


# -------------------------------------------------------------------- #
# vote_on_proposal — own vote and inbound vote
# -------------------------------------------------------------------- #


def test_vote_on_proposal_success(three_node):
    alpha = three_node["alpha"]
    beta = three_node["beta"]
    p = alpha.propose_state_change({"action": "deploy"})

    # Beta must ingest the proposal before voting
    beta.ingest_proposal(p)

    # Alpha casts its own vote
    v_alpha = alpha.vote_on_proposal(p.proposal_id, VoteDelta.YES)
    assert v_alpha.delta == VoteDelta.YES

    # Beta records a vote on alpha's proposal
    v_beta = beta.vote_on_proposal(p.proposal_id, VoteDelta.YES)
    # Beta's local graph should have the edge beta -> alpha (originator)
    assert ("beta", "alpha") in beta.graph_edges()


def test_double_vote_same_delta_is_idempotent(three_node):
    alpha = three_node["alpha"]
    p = alpha.propose_state_change({"x": 1})
    v1 = alpha.vote_on_proposal(p.proposal_id, VoteDelta.YES)
    v2 = alpha.vote_on_proposal(p.proposal_id, VoteDelta.YES)
    assert v1 == v2


def test_double_vote_conflicting_delta_raises_byzantine(three_node):
    alpha = three_node["alpha"]
    p = alpha.propose_state_change({"x": 1})
    alpha.vote_on_proposal(p.proposal_id, VoteDelta.YES)
    with pytest.raises(ByzantineFault, match="double-voted"):
        alpha.vote_on_proposal(p.proposal_id, VoteDelta.NO)


def test_vote_on_unknown_proposal_raises(three_node):
    alpha = three_node["alpha"]
    with pytest.raises(ConsensusError, match="Unknown proposal"):
        alpha.vote_on_proposal("prop-123", VoteDelta.YES)


def test_vote_bad_signature_raises_byzantine(three_node):
    alpha = three_node["alpha"]
    beta = three_node["beta"]
    p = alpha.propose_state_change({"x": 1})
    beta.ingest_proposal(p)

    # Forge a vote with an invalid signature
    forged = Vote(
        proposal_id=p.proposal_id,
        node_id="beta",
        delta=VoteDelta.YES,
        vector_clock=(0, 0, 0),
        timestamp=0.0,
        signature="bad-sig",
    )

    # When alpha receives the forged vote from beta, signature verification fails
    with pytest.raises(ByzantineFault, match="Invalid signature"):
        alpha.vote_on_proposal(
            p.proposal_id, VoteDelta.YES, from_node="beta", inbound_vote=forged
        )

    # A valid vote from beta should succeed
    valid = beta.vote_on_proposal(p.proposal_id, VoteDelta.YES)
    replay = alpha.vote_on_proposal(
        p.proposal_id, VoteDelta.YES, from_node="beta", inbound_vote=valid
    )
    assert replay == valid


# -------------------------------------------------------------------- #
# commit_if_quorum — happy path
# -------------------------------------------------------------------- #


def test_commit_reaches_quorum(three_node):
    alpha = three_node["alpha"]
    beta = three_node["beta"]
    gamma = three_node["gamma"]

    p = alpha.propose_state_change({"deploy": True})

    # Need 2f+1 = 3 votes in a 4-node fleet with f=1
    alpha.vote_on_proposal(p.proposal_id, VoteDelta.YES)
    beta.ingest_proposal(p)
    beta_vote = beta.vote_on_proposal(p.proposal_id, VoteDelta.YES)
    gamma.ingest_proposal(p)
    gamma_vote = gamma.vote_on_proposal(p.proposal_id, VoteDelta.YES)

    # Simulate network delivery of votes to alpha (records edges in graph)
    alpha.vote_on_proposal(
        p.proposal_id, beta_vote.delta, from_node="beta", inbound_vote=beta_vote
    )
    alpha.vote_on_proposal(
        p.proposal_id, gamma_vote.delta, from_node="gamma", inbound_vote=gamma_vote
    )

    assert alpha.commit_if_quorum(p.proposal_id) is True


def test_commit_not_enough_votes_returns_false(three_node):
    alpha = three_node["alpha"]
    p = alpha.propose_state_change({"deploy": True})
    alpha.vote_on_proposal(p.proposal_id, VoteDelta.YES)
    assert alpha.commit_if_quorum(p.proposal_id) is False


# -------------------------------------------------------------------- #
# commit_if_quorum — partition tolerance
# -------------------------------------------------------------------- #


def test_partition_fault_when_graph_too_fractured(four_node):
    # n=4, f=1, quorum=3
    n1 = four_node["n1"]
    n2 = four_node["n2"]
    n3 = four_node["n3"]
    # n4 is partitioned away

    p = n1.propose_state_change({"x": 1})
    n1.vote_on_proposal(p.proposal_id, VoteDelta.YES)
    n2.ingest_proposal(p)
    n2_vote = n2.vote_on_proposal(p.proposal_id, VoteDelta.YES)
    n3.ingest_proposal(p)
    n3_vote = n3.vote_on_proposal(p.proposal_id, VoteDelta.YES)

    # Deliver n2's vote to n1 (so n1 sees the edge n2→n1)
    n1.vote_on_proposal(
        p.proposal_id, n2_vote.delta, from_node="n2", inbound_vote=n2_vote
    )

    # n3's vote is dropped (partition).  n1 only knows about n1 and n2.
    # However, the active voters are 2 < quorum=3, so we should NOT raise
    # PartitionFault yet — we just return False.  To trigger PartitionFault,
    # we need n1 to know about n3's vote EXISTENCE but not its graph edge.
    # We simulate this by injecting n3's vote into n1's _votes directly
    # (so n1 knows n3 voted) but NOT adding the graph edge.
    n1._votes.setdefault(p.proposal_id, []).append(n3_vote)

    with pytest.raises(PartitionFault, match="Largest connected component"):
        n1.commit_if_quorum(p.proposal_id)


# -------------------------------------------------------------------- #
# H¹ holonomy / emergence detection
# -------------------------------------------------------------------- #


def test_detect_emergence_finds_trivial_cycle(three_node):
    alpha = three_node["alpha"]
    beta = three_node["beta"]
    gamma = three_node["gamma"]

    p = alpha.propose_state_change({"x": 1})
    alpha.vote_on_proposal(p.proposal_id, VoteDelta.YES)
    beta.ingest_proposal(p)
    beta_vote = beta.vote_on_proposal(p.proposal_id, VoteDelta.NO)
    gamma.ingest_proposal(p)
    gamma_vote = gamma.vote_on_proposal(p.proposal_id, VoteDelta.ABSTAIN)

    # Simulate alpha has received all votes
    alpha._votes.setdefault(p.proposal_id, []).extend([beta_vote, gamma_vote])

    # Build edges so a cycle alpha→beta→gamma→alpha exists
    alpha._vote_graph.setdefault("beta", set()).add("alpha")
    alpha._vote_graph.setdefault("gamma", set()).add("beta")
    alpha._vote_graph.setdefault("alpha", set()).add("gamma")

    cycles = alpha.detect_emergence(p.proposal_id)
    assert len(cycles) > 0
    # At least one cycle should be trivial (YES + NO + ABSTAIN = 1-1+0 = 0)
    trivial = [c for c in cycles if c.is_trivial()]
    assert len(trivial) > 0


def test_detect_emergence_non_trivial_cycle(three_node):
    alpha = three_node["alpha"]
    beta = three_node["beta"]
    gamma = three_node["gamma"]

    p = alpha.propose_state_change({"x": 1})
    alpha.vote_on_proposal(p.proposal_id, VoteDelta.YES)
    beta.ingest_proposal(p)
    beta_vote = beta.vote_on_proposal(p.proposal_id, VoteDelta.YES)
    gamma.ingest_proposal(p)
    gamma_vote = gamma.vote_on_proposal(p.proposal_id, VoteDelta.YES)

    # Simulate alpha has received all votes
    alpha._votes.setdefault(p.proposal_id, []).extend([beta_vote, gamma_vote])

    # Build a cycle alpha→beta→gamma→alpha
    alpha._vote_graph.setdefault("beta", set()).add("alpha")
    alpha._vote_graph.setdefault("gamma", set()).add("beta")
    alpha._vote_graph.setdefault("alpha", set()).add("gamma")

    cycles = alpha.detect_emergence(p.proposal_id)
    # YES + YES + YES = 3 → non-trivial holonomy (all agreeing on a cycle)
    non_trivial = [c for c in cycles if not c.is_trivial()]
    assert len(non_trivial) > 0
    assert all(c.accumulated_delta != 0 for c in non_trivial)


def test_commit_raises_emergence_on_non_trivial_holonomy(three_node):
    alpha = three_node["alpha"]
    beta = three_node["beta"]
    gamma = three_node["gamma"]

    p = alpha.propose_state_change({"x": 1})
    alpha.vote_on_proposal(p.proposal_id, VoteDelta.YES)
    beta.ingest_proposal(p)
    beta_vote = beta.vote_on_proposal(p.proposal_id, VoteDelta.YES)
    gamma.ingest_proposal(p)
    gamma_vote = gamma.vote_on_proposal(p.proposal_id, VoteDelta.YES)

    # Simulate alpha has received all votes
    alpha._votes.setdefault(p.proposal_id, []).extend([beta_vote, gamma_vote])

    # Inject edges to create a non-trivial cycle
    alpha._vote_graph.setdefault("beta", set()).add("alpha")
    alpha._vote_graph.setdefault("gamma", set()).add("beta")
    alpha._vote_graph.setdefault("alpha", set()).add("gamma")

    with pytest.raises(EmergenceDetected, match="Non-trivial holonomy"):
        alpha.commit_if_quorum(p.proposal_id)


# -------------------------------------------------------------------- #
# Integration — full round with cross-node vote sharing
# -------------------------------------------------------------------- #


def test_full_round_4node_no_partition(four_node):
    # n=4, f=1, quorum=3
    nodes = four_node
    p = nodes["n1"].propose_state_change({"shard": "move"})

    # Phase 1: every node ingests the proposal
    for n in nodes.values():
        if n.node_id != p.node_id:
            n.ingest_proposal(p)

    # Phase 2: every node votes YES and "gossips" its vote to all others
    for n in nodes.values():
        v = n.vote_on_proposal(p.proposal_id, VoteDelta.YES)
        for peer in nodes.values():
            if peer.node_id != n.node_id:
                # Simulate network delivery of the signed vote
                peer.vote_on_proposal(
                    p.proposal_id, v.delta, from_node=n.node_id, inbound_vote=v
                )

    assert nodes["n1"].commit_if_quorum(p.proposal_id) is True
    assert nodes["n2"].commit_if_quorum(p.proposal_id) is True


# -------------------------------------------------------------------- #
# Edge cases
# -------------------------------------------------------------------- #


def test_empty_state_delta_is_valid(three_node):
    alpha = three_node["alpha"]
    p = alpha.propose_state_change({})
    assert p.state_delta == {}


def test_abstain_votes_dont_count_for_quorum_but_dont_break(three_node):
    alpha = three_node["alpha"]
    beta = three_node["beta"]
    gamma = three_node["gamma"]

    p = alpha.propose_state_change({"x": 1})
    alpha.vote_on_proposal(p.proposal_id, VoteDelta.YES)
    beta.ingest_proposal(p)
    beta.vote_on_proposal(p.proposal_id, VoteDelta.ABSTAIN)
    gamma.ingest_proposal(p)
    gamma.vote_on_proposal(p.proposal_id, VoteDelta.YES)

    # YES votes = 2, quorum = 3 → not committed yet
    assert alpha.commit_if_quorum(p.proposal_id) is False


def test_detect_emergence_on_unknown_proposal_raises(three_node):
    alpha = three_node["alpha"]
    with pytest.raises(ConsensusError, match="Unknown proposal"):
        alpha.detect_emergence("prop-ghost")


def test_holonomy_cycle_repr_is_sane():
    c = HolonomyCycle(nodes=("a", "b", "c"), accumulated_delta=2)
    assert not c.is_trivial()
    c2 = HolonomyCycle(nodes=("a", "b", "c"), accumulated_delta=0)
    assert c2.is_trivial()
