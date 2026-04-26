"""
Convert a networkx node-link JSON graph (string IDs, rich metadata) into
the karate-shaped fixture format used by the cross-validation harness:
integer IDs 0..n-1, undirected edge list, optional weights as [u, v, w].

Bridges arbitrary networkx-emitted node-link graphs (e.g. extracted
dependency graphs, social networks, citation networks) into the integer-
id schema that bench/compare/ref/run_leiden.py and the TS test suite
both consume.

Determinism: string IDs are sorted lexicographically before being assigned
integer IDs. The mapping is therefore reproducible across machines.

Any pre-existing `community` field on nodes is carried through as
`prior_communities` (integer-id-aligned). It is NOT ground truth — it is
an externally-computed clustering, intended for diagnostic emit only.

Usage:
    python ref/convert_nodelink.py \
        --input /path/to/some-graph.json \
        --name some-graph \
        --source "free-form provenance string committed in the fixture" \
        --out fixtures/some-graph.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def convert(node_link: dict, name: str, source: str) -> dict:
    if node_link.get("directed"):
        sys.exit("converter expects undirected graphs; got directed=True")
    if node_link.get("multigraph"):
        sys.exit("converter expects simple graphs; got multigraph=True")

    nodes_in = node_link["nodes"]
    links_in = node_link["links"]

    string_ids = [n["id"] for n in nodes_in]
    if len(set(string_ids)) != len(string_ids):
        sys.exit("duplicate node IDs in input")

    sorted_ids = sorted(string_ids)
    id_to_int = {sid: i for i, sid in enumerate(sorted_ids)}

    by_id = {n["id"]: n for n in nodes_in}
    prior_communities = [by_id[sid].get("community", -1) for sid in sorted_ids]

    edges_out: list[list] = []
    seen: set[tuple[int, int]] = set()
    for e in links_in:
        u_str, v_str = e["source"], e["target"]
        if u_str == v_str:
            continue  # drop self-loops
        u, v = id_to_int[u_str], id_to_int[v_str]
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in seen:
            continue  # drop duplicate undirected edges
        seen.add((a, b))
        w = e.get("weight", 1.0)
        if w == 1.0:
            edges_out.append([a, b])
        else:
            edges_out.append([a, b, w])

    edges_out.sort(key=lambda t: (t[0], t[1]))

    return {
        "name": name,
        "source": source,
        "n": len(sorted_ids),
        "directed": False,
        "edges": edges_out,
        "prior_communities": prior_communities,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="path to node-link JSON")
    parser.add_argument("--name", required=True,
                        help="fixture name (e.g. 'httpx')")
    parser.add_argument("--source", required=True,
                        help="provenance string for the fixture")
    parser.add_argument("--out", required=True,
                        help="output path (e.g. fixtures/httpx.json)")
    args = parser.parse_args()

    src_path = pathlib.Path(args.input).resolve()
    out_path = pathlib.Path(args.out).resolve()

    if not src_path.exists():
        sys.exit(f"input not found: {src_path}")

    node_link = json.loads(src_path.read_text())
    fixture = convert(node_link, args.name, args.source)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fixture, indent=2) + "\n")

    n = fixture["n"]
    m = len(fixture["edges"])
    weighted = sum(1 for e in fixture["edges"] if len(e) == 3)
    n_prior = len({c for c in fixture["prior_communities"] if c >= 0})
    print(
        f"wrote {out_path.name}: n={n}, m={m}, weighted_edges={weighted}, "
        f"prior_communities={n_prior}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
