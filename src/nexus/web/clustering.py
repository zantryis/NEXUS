"""Thread clustering by entity overlap for the thread list display."""


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cluster_threads(threads: list[dict], threshold: float = 0.35) -> list[list[dict]]:
    """Group threads by key_entity overlap using union-find.

    Returns clusters sorted by max significance (desc). Each cluster
    is internally sorted by significance (desc).
    """
    n = len(threads)
    if n == 0:
        return []

    # Build entity sets
    entity_sets = []
    for t in threads:
        entities = t.get("key_entities") or []
        entity_sets.append({e.lower() for e in entities})

    # Union-find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Pairwise comparison
    for i in range(n):
        for j in range(i + 1, n):
            if _jaccard(entity_sets[i], entity_sets[j]) >= threshold:
                union(i, j)

    # Group by root
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    # Build result clusters
    clusters = []
    for indices in groups.values():
        cluster = [threads[i] for i in indices]
        cluster.sort(key=lambda t: t.get("significance", 0), reverse=True)
        clusters.append(cluster)

    # Sort clusters by max significance
    clusters.sort(key=lambda c: c[0].get("significance", 0), reverse=True)
    return clusters
