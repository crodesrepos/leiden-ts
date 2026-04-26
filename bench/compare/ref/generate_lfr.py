"""
Generate LFR (Lancichinetti-Fortunato-Radicchi 2008) benchmark graphs and
write them as JSON fixtures with ground-truth community assignments.

LFR is the canonical synthetic benchmark for community-detection algorithms
because:
    1. The mixing parameter μ controls how strong the community structure is
       (μ=0 → perfect communities; μ=1 → random graph).
    2. Each node has a known ground-truth community assignment, so we can
       compute NMI and ARI against truth (not just against another impl).
    3. Sizes scale freely (we generate at 1k and 10k for parity benchmarks).

We seed deterministically and commit the output JSON, so cross-validation
runs are reproducible regardless of networkx version drift.

Usage:
    python ref/generate_lfr.py
"""

from __future__ import annotations

import json
import pathlib
import sys

try:
    import networkx as nx
except ImportError:
    sys.exit("Install reqs first: pip install -r requirements.txt")


HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = HERE.parent / "fixtures"


def _generate(n: int, mu: float, seed: int, label: str) -> dict:
    """LFR with default parameters tuned for community-detection literature."""
    # Conventional LFR parameters (Lancichinetti et al. 2008):
    #   tau1: power-law exponent for degree distribution (typically 2-3)
    #   tau2: power-law exponent for community size (typically 1-2)
    #   average_degree, max_degree: scale with n
    avg_deg = max(10, int(n / 100))
    max_deg = max(50, int(n / 20))
    min_comm = max(10, int(n / 50))
    max_comm = max(50, int(n / 10))

    G = nx.LFR_benchmark_graph(
        n=n,
        tau1=3.0,
        tau2=1.5,
        mu=mu,
        average_degree=avg_deg,
        max_degree=max_deg,
        min_community=min_comm,
        max_community=max_comm,
        seed=seed,
        max_iters=1000,
    )

    # Strip self-loops (LFR can produce them; our pipeline doesn't accept).
    G.remove_edges_from(nx.selfloop_edges(G))

    # Build ground-truth community assignments.
    assignments = [-1] * G.number_of_nodes()
    community_index: dict[frozenset[int], int] = {}
    for u in G.nodes:
        community = frozenset(G.nodes[u]["community"])
        cid = community_index.setdefault(community, len(community_index))
        assignments[u] = cid

    # Build edge list (canonical ordering — lo, hi).
    edges: list[list[int]] = []
    for u, v in G.edges():
        if u == v:
            continue
        lo, hi = (u, v) if u < v else (v, u)
        edges.append([lo, hi])
    edges.sort()

    return {
        "name": label,
        "source": (
            "Lancichinetti, A., Fortunato, S., & Radicchi, F. (2008). "
            "Benchmark graphs for testing community detection algorithms. "
            "Physical Review E, 78, 046110. Generated via "
            f"networkx.LFR_benchmark_graph (n={n}, μ={mu}, seed={seed})."
        ),
        "n": G.number_of_nodes(),
        "directed": False,
        "mixing_parameter_mu": mu,
        "lfr_seed": seed,
        "edges": edges,
        "ground_truth_assignments": assignments,
        "ground_truth_n_communities": len(community_index),
    }


def main() -> int:
    targets = [
        ("lfr-1k-mu03", 1000, 0.3, 42),
        ("lfr-1k-mu05", 1000, 0.5, 42),
        ("lfr-10k-mu03", 10000, 0.3, 42),
    ]
    for name, n, mu, seed in targets:
        print(f"Generating {name} (n={n}, μ={mu}, seed={seed})...", file=sys.stderr)
        try:
            data = _generate(n, mu, seed, name)
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            continue
        path = FIXTURES / f"{name}.json"
        path.write_text(json.dumps(data, separators=(",", ":")) + "\n")
        print(
            f"  wrote {path.name}: n={data['n']} m={len(data['edges'])} "
            f"communities={data['ground_truth_n_communities']}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
