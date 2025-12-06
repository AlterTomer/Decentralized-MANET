import torch
from collections import deque

def find_multicast_subgraphs(
    adj_matrix,
    src: int,
    receivers: list[int],
    max_solutions: int | None = None,
    prune_unreachable: bool = True,
    undirected: bool = True,
    enforce_minimality: bool = True,
):
    """
    Enumerate all directed (or undirected) subgraphes that connect `src` to every node in `receivers`.
    Each returned subgraph is an [n, n] boolean tensor mask. If `undirected=True`, the mask is symmetric
    and an edge {u,v} is represented by mask[u,v]=mask[v,u]=True. The function performs a minimality
    reduction pass so that no returned subgraph contains a removable edge (unless enforce_minimality=False).

    Args:
        adj_matrix: [n, n] torch.Tensor (0/1 or bool). Edge i->j exists if adj[i, j] != 0.
        src: int, source node index.
        receivers: list[int], destination node indices to be connected to `src`.
        max_solutions: optional cap on number of subgraphes returned.
        prune_unreachable: use base-graph reachability to prune dead branches early.
        undirected: treat the input topology as undirected (symmetrize for search/connectivity).
        enforce_minimality: after a candidate subgraph is found, iteratively remove any edge
            whose removal preserves connectivity (src reaches all receivers) in that subgraph.

    Returns:
        List[torch.Tensor]: list of [n, n] boolean masks, each a (directional) connecting subgraph.
                            If undirected=True, masks are symmetric.
    """

    n = int(adj_matrix.size(0))
    dev = adj_matrix.device
    adj_bool = (adj_matrix != 0)

    if undirected:
        adj_bool = adj_bool | adj_bool.t()  # symmetrize for search/connectivity

    src = int(src)
    if isinstance(receivers[0], list) and len(receivers) == 1:
        receivers = receivers[0]
    receivers = [int(r) for r in receivers]

    # ---------- helpers ----------
    def bfs_reaches_all(mask_bool: torch.Tensor) -> bool:
        """Check if src reaches all receivers within the given subgraph mask (respecting 'undirected')."""
        seen = torch.zeros(n, dtype=torch.bool, device=dev)
        q = deque([src])
        seen[src] = True
        while q:
            u = q.popleft()
            nbrs = (mask_bool[u]).nonzero(as_tuple=True)[0]
            for v in nbrs.tolist():
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        return all(bool(seen[r].item()) for r in receivers)

    def canonicalize_edges(edges: list[tuple[int, int]]) -> frozenset[tuple[int, int]]:
        """Canonical edge set representation for deduplication (sorted pairs if undirected)."""
        if undirected:
            return frozenset((min(u, v), max(u, v)) for (u, v) in edges)
        else:
            return frozenset(edges)

    def mask_from_edges(edges_set: frozenset[tuple[int, int]]) -> torch.Tensor:
        """Build [n,n] bool mask from canonical edge set."""
        mask = torch.zeros((n, n), dtype=torch.bool, device=dev)
        for (u, v) in edges_set:
            if undirected:
                mask[u, v] = True
                mask[v, u] = True
            else:
                mask[u, v] = True
        return mask

    # Optional base-graph reachability for pruning (on the original adjacency)
    if prune_unreachable:
        adj_i = adj_bool.to(torch.float32)
        eye_i = torch.eye(n, dtype=torch.int32, device=dev)
        reach_b = (adj_i + eye_i) > 0
        for _ in range(n - 1):
            nxt_b = (reach_b.to(torch.float32) @ adj_i) > 0
            new_b = reach_b | nxt_b
            if new_b.equal(reach_b):
                break
            reach_b = new_b

        def any_from_R_can_reach(t: int, R_nodes: set[int]) -> bool:
            if not R_nodes:
                return False
            idx = torch.tensor(sorted(R_nodes), device=dev)
            return bool(reach_b[idx, t].any().item())
    else:
        def any_from_R_can_reach(t: int, R_nodes: set[int]) -> bool:
            return True

    solutions: list[torch.Tensor] = []
    seen_sets: set[frozenset[tuple[int, int]]] = set()

    # ---------- backtracking over frontier expansions ----------
    # We grow reachable set R by adding edges (u->v) with u in R and v not in R.
    # For undirected mode, adjacency is symmetric and the same logic applies.
    def backtrack(R_nodes: set[int], edges: list[tuple[int, int]]):
        if max_solutions is not None and len(solutions) >= max_solutions:
            return

        # If all receivers are covered, emit (after optional minimality reduction)
        if all(r in R_nodes for r in receivers):
            E_set = canonicalize_edges(edges)

            # Minimality reduction: try removing each edge if connectivity is preserved.
            if enforce_minimality:
                changed = True
                while changed:
                    changed = False
                    current_edges = list(E_set)
                    for e in current_edges:
                        # remove e (and its symmetric if undirected via canonical set)
                        tmp_set = set(E_set)
                        tmp_set.remove(e)
                        mask_tmp = mask_from_edges(frozenset(tmp_set))
                        if bfs_reaches_all(mask_tmp):
                            E_set = frozenset(tmp_set)
                            changed = True
                            break  # restart scan after change

            if E_set not in seen_sets:
                seen_sets.add(E_set)
                solutions.append(mask_from_edges(E_set))
            return

        # Prune dead branches: every uncovered receiver must be reachable from some node in R (in base graph)
        if prune_unreachable:
            uncovered = [r for r in receivers if r not in R_nodes]
            if any(not any_from_R_can_reach(t, R_nodes) for t in uncovered):
                return

        # Frontier: edges u->v with u in R and v not in R
        for u in sorted(R_nodes):
            vs = (adj_bool[u]).nonzero(as_tuple=True)[0].tolist()
            for v in vs:
                if v in R_nodes:
                    continue
                edges.append((u, v))
                R_nodes.add(v)
                backtrack(R_nodes, edges)
                R_nodes.remove(v)
                edges.pop()

    backtrack(R_nodes={src}, edges=[])
    return solutions


def mask_to_edges(mask: torch.Tensor):
    n = mask.size(0)
    return [(int(i), int(j)) for i in range(n) for j in range(n) if bool(mask[i, j])]
