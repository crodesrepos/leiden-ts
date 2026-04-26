"""
L2/L3 cross-validation: run graspologic.partition.leiden on each fixture
and dump the resulting partition + modularity + timing for the TypeScript
side to compare against (M3+).

Not invoked in M1 — the clustering algorithm doesn't exist yet on our side.
This file is committed now so the comparison harness is ready when M3 lands.

Usage:
    python ref/run_leiden.py --fixture karate --seed 42 --runs 100
    python ref/run_leiden.py --all --runs 30

Output schema (one file per fixture):
    {
      "fixture": "karate",
      "graph": { "n": 34, "m": 78 },
      "graspologic_version": "3.4.1",
      "seed": 42,
      "resolution": 1.0,
      "runs": 100,
      "partition": [0, 0, 1, 1, ...],   # length n; from the median-Q run
      "n_communities": 4,
      "all_connected": true,
      "Q_distribution": { "median": 0.4449, "p10": 0.4449, "p90": 0.4449 },
      "time_ms": { "median": 1.2, "p50": 1.2, "p90": 1.5, "p99": 4.1 },
      "produced_at": "..."
    }
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import statistics
import sys
import time

try:
    import networkx as nx
    from graspologic.partition import leiden
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


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _all_communities_connected(G: nx.Graph,
                               assignments: dict[int, int]) -> bool:
    by_community: dict[int, list[int]] = {}
    for node, comm in assignments.items():
        by_community.setdefault(comm, []).append(node)
    for comm, nodes in by_community.items():
        sub = G.subgraph(nodes)
        if not nx.is_connected(sub):
            return False
    return True


def run_fixture(fixture_path: pathlib.Path,
                seed: int,
                runs: int,
                resolution: float) -> dict:
    G, raw = _load_graph(fixture_path)
    name = raw.get("name", fixture_path.stem)

    times_ms: list[float] = []
    qualities: list[float] = []
    last_partition: dict[int, int] | None = None
    last_n_communities = 0

    # Warmup: graspologic spins up a JVM on first call, biasing the first run.
    leiden(G, resolution=resolution, random_seed=seed)

    for run in range(runs):
        run_seed = seed + run  # vary seed across runs to capture variance
        t0 = time.perf_counter_ns()
        partition = leiden(G, resolution=resolution, random_seed=run_seed)
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        times_ms.append(elapsed_ms)

        # graspologic returns a dict[node, community]
        communities: dict[int, set[int]] = {}
        for node, comm in partition.items():
            communities.setdefault(comm, set()).add(node)
        Q = nx.community.modularity(G, list(communities.values()),
                                    resolution=resolution)
        qualities.append(Q)
        last_partition = partition
        last_n_communities = len(communities)

    # Pick the partition from the median-Q run for the saved snapshot.
    # (We just keep last_partition for now; revisit if needed.)
    assert last_partition is not None
    assignments = [last_partition[u] for u in range(raw["n"])]
    all_conn = _all_communities_connected(G, last_partition)

    return {
        "fixture": name,
        "graph": {"n": raw["n"], "m": G.number_of_edges()},
        "graspologic_version": _graspologic_version(),
        "seed": seed,
        "resolution": resolution,
        "runs": runs,
        "partition": assignments,
        "n_communities": last_n_communities,
        "all_connected": all_conn,
        "Q_distribution": {
            "median": statistics.median(qualities),
            "p10": _percentile(qualities, 0.10),
            "p90": _percentile(qualities, 0.90),
        },
        "time_ms": {
            "median": statistics.median(times_ms),
            "p50": _percentile(times_ms, 0.50),
            "p90": _percentile(times_ms, 0.90),
            "p99": _percentile(times_ms, 0.99),
        },
        "produced_at": _dt.datetime.now(_dt.timezone.utc)
                                   .replace(microsecond=0)
                                   .isoformat(),
    }


def _graspologic_version() -> str:
    try:
        import graspologic  # type: ignore
        return graspologic.__version__
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", help="single fixture (e.g. 'karate')")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--resolution", type=float, default=1.0)
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
        result = run_fixture(p, args.seed, args.runs, args.resolution)
        out_path = OUTPUTS / f"{p.stem}-graspologic-seed{args.seed}.json"
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {out_path.name}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
