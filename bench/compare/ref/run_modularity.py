"""
L1 cross-validation: dump networkx.community.modularity values for fixed
partitions on each fixture graph. Produces JSON files in `outputs/` that
the TypeScript test suite reads as reference values.

Usage:
    python ref/run_modularity.py --fixture karate
    python ref/run_modularity.py --all

Output schema (one file per fixture):
    {
      "fixture": "karate",
      "graph": { "n": ..., "m": ..., "directed": false },
      "networkx_version": "3.3",
      "partitions": {
        "ground_truth_2_partition": {
          "Q": 0.37179487179487175,
          "Q_resolution_2": 0.18...,
          "n_communities": 2
        },
        "all_singletons": { "Q": -0.05..., "n_communities": 34 },
        "all_one_community": { "Q": 0.0, "n_communities": 1 }
      },
      "produced_at": "2026-04-25T..."
    }
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
    """Load a fixture and return (graph, raw_dict)."""
    raw = json.loads(fixture_path.read_text())
    G = nx.Graph()
    G.add_nodes_from(range(raw["n"]))
    for edge in raw["edges"]:
        u, v = edge[0], edge[1]
        w = edge[2] if len(edge) > 2 else 1.0
        G.add_edge(u, v, weight=w)
    return G, raw


def _partition_from_assignments(assignments: list[int]) -> list[set[int]]:
    """Convert an assignment array into a list of community sets."""
    by_community: dict[int, set[int]] = {}
    for node, comm in enumerate(assignments):
        by_community.setdefault(comm, set()).add(node)
    return list(by_community.values())


def _modularity_block(G: nx.Graph, communities: list[set[int]],
                      resolutions=(1.0, 2.0)) -> dict:
    out = {"n_communities": len(communities)}
    for gamma in resolutions:
        Q = nx.community.modularity(G, communities, resolution=gamma)
        suffix = "" if gamma == 1.0 else f"_resolution_{gamma}"
        out[f"Q{suffix}"] = Q
    return out


def run_fixture(fixture_path: pathlib.Path) -> dict:
    G, raw = _load_graph(fixture_path)
    name = raw.get("name", fixture_path.stem)

    partitions: dict[str, dict] = {}

    # Ground-truth 2-partition (when fixture provides one).
    if "ground_truth_2_partition_assignments" in raw:
        comms = _partition_from_assignments(
            raw["ground_truth_2_partition_assignments"]
        )
        partitions["ground_truth_2_partition"] = _modularity_block(G, comms)

    # All singletons — every node its own community. Modularity is
    # -Σ (kᵤ/2m)² (negative for any non-empty graph).
    singletons = [{u} for u in range(raw["n"])]
    partitions["all_singletons"] = _modularity_block(G, singletons)

    # All-one — single community contains every node. Modularity is 0
    # because Σᵢₙ/2m = 1 and (Σₜₒₜ/2m)² = 1.
    if raw["n"] > 0:
        all_one = [set(range(raw["n"]))]
        partitions["all_one_community"] = _modularity_block(G, all_one)

    return {
        "fixture": name,
        "graph": {
            "n": raw["n"],
            "m": G.number_of_edges(),
            "directed": False,
        },
        "networkx_version": nx.__version__,
        "partitions": partitions,
        "produced_at": _dt.datetime.now(_dt.timezone.utc)
                                  .replace(microsecond=0)
                                  .isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", help="single fixture to run (e.g. 'karate')")
    parser.add_argument("--all", action="store_true",
                        help="run every fixture in fixtures/")
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
        result = run_fixture(p)
        out_path = OUTPUTS / f"{p.stem}-modularity.json"
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {out_path.relative_to(HERE.parent.parent.parent)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
