"""
L2-prep cross-validation: run networkx.community.louvain_communities on each
fixture and dump the partition, modularity, community count, and per-community
connectivity. Used by the TypeScript test suite to validate that leiden-ts'
clustering output is in the same ballpark as a known-good Python implementation.

Note on the Leiden gap:
    networkx ships Louvain only (no Leiden as of 3.6.x). graspologic has Leiden
    but is a heavy install. For an L2-equivalent pass at this milestone we use
    Louvain as the baseline. Louvain ≡ "localMove + aggregate iterated to
    convergence" — i.e., Leiden Phase 1 + Phase 3, missing Phase 2 (refine).

    Our M2+M3 output (= Phase 1 + Phase 2, missing Phase 3) and networkx
    Louvain (= Phase 1 + Phase 3, missing Phase 2) are NOT directly comparable
    on Q. They share Phase 1 but diverge after.

    What IS comparable:
      - Q after our localMove (= our Phase 1) vs Q at the FIRST level of
        networkx Louvain. nx doesn't expose this directly, but we can
        approximate it by running our modularity() on Louvain's final
        partition projected back to the original graph.
      - Connectivity: every output community of refine() is connected
        (paper Theorem 1). networkx Louvain output may have disconnected
        communities — that's the bug Leiden was designed to fix.

    Q values:
      - Our localMove on Karate: ~0.347 (local optimum at original-graph level)
      - Networkx Louvain on Karate: ~0.444 (full converged hierarchy)
    The ~0.10 gap is exactly what M4 (aggregation) closes.

Usage:
    python ref/run_louvain.py --fixture karate
    python ref/run_louvain.py --all
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import sys

try:
    import networkx as nx
except ImportError:
    sys.exit("Install reqs first: pip install -r requirements.txt")


HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = HERE.parent / "fixtures"
OUTPUTS = HERE / "outputs"
OUTPUTS.mkdir(exist_ok=True)


def _load_graph(fixture_path: pathlib.Path) -> tuple[nx.Graph, dict]:
    raw = json.loads(fixture_path.read_text())
    G = nx.Graph()
    G.add_nodes_from(range(raw["n"]))
    for edge in raw["edges"]:
        u, v = edge[0], edge[1]
        w = edge[2] if len(edge) > 2 else 1.0
        G.add_edge(u, v, weight=w)
    return G, raw


def _all_communities_connected(
    G: nx.Graph, communities: list[set[int]]
) -> bool:
    """Verify every community induces a connected subgraph of G."""
    for nodes in communities:
        sub = G.subgraph(nodes)
        if not nx.is_connected(sub):
            return False
    return True


def _disconnected_count(G: nx.Graph, communities: list[set[int]]) -> int:
    """Count of communities that are NOT internally connected."""
    n = 0
    for nodes in communities:
        sub = G.subgraph(nodes)
        if not nx.is_connected(sub):
            n += 1
    return n


def run_fixture(fixture_path: pathlib.Path, seed: int) -> dict:
    G, raw = _load_graph(fixture_path)
    n = raw["n"]

    # networkx Louvain: returns list of sets, one per community.
    communities = nx.community.louvain_communities(
        G,
        weight="weight",
        resolution=1.0,
        threshold=1e-7,
        seed=seed,
    )

    # Build assignments array.
    assignments = [-1] * n
    for cid, nodes in enumerate(communities):
        for u in nodes:
            assignments[u] = cid

    Q = nx.community.modularity(G, communities, resolution=1.0)
    all_connected = _all_communities_connected(G, communities)
    disconnected = _disconnected_count(G, communities)

    return {
        "fixture": raw.get("name", fixture_path.stem),
        "graph": {"n": n, "m": G.number_of_edges()},
        "networkx_version": nx.__version__,
        "algorithm": "louvain",
        "seed": seed,
        "resolution": 1.0,
        "n_communities": len(communities),
        "assignments": assignments,
        "modularity": Q,
        "all_connected": all_connected,
        "disconnected_community_count": disconnected,
        "produced_at": _dt.datetime.now(_dt.timezone.utc)
                                   .replace(microsecond=0)
                                   .isoformat(),
        "_note": (
            "Louvain reference. Our M2+M3 (localMove + refine) is NOT "
            "directly comparable on Q because we are missing aggregation. "
            "The connectivity property IS comparable: every output community "
            "of refine() must be connected (paper Theorem 1)."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", help="single fixture (e.g. 'karate')")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.fixture and not args.all:
        parser.error("either --fixture or --all is required")

    paths: list[pathlib.Path]
    if args.all:
        paths = sorted(FIXTURES.glob("*.json"))
    else:
        path = FIXTURES / f"{args.fixture}.json"
        if not path.exists():
            sys.exit(f"fixture not found: {path}")
        paths = [path]

    for p in paths:
        result = run_fixture(p, args.seed)
        out_path = OUTPUTS / f"{p.stem}-louvain-seed{args.seed}.json"
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {out_path.name}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
