"""SVG radial graph generator for entity relationship visualization."""

import math


def render_entity_network_svg(
    center_name: str,
    related: list[dict],
    width: int = 480,
    height: int = 480,
) -> str:
    """Generate an inline SVG showing entity relationships in a radial layout.

    Args:
        center_name: Name of the central entity.
        related: List of dicts with 'id', 'canonical_name', 'co_occurrence_count'.
        width/height: SVG dimensions.

    Returns:
        SVG markup string.
    """
    if not related:
        return ""

    cx, cy = width / 2, height / 2
    radius = min(width, height) / 2 - 60
    max_co = max(r.get("co_occurrence_count", 1) for r in related) or 1

    # Limit to top 12 related entities
    nodes = sorted(related, key=lambda r: r.get("co_occurrence_count", 0), reverse=True)[:12]

    lines = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
    ]

    # Draw edges
    for i, node in enumerate(nodes):
        angle = (2 * math.pi * i) / len(nodes) - math.pi / 2
        nx = cx + radius * math.cos(angle)
        ny = cy + radius * math.sin(angle)
        strength = node.get("co_occurrence_count", 1) / max_co
        opacity = 0.15 + 0.45 * strength
        lines.append(
            f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" '
            f'class="graph-edge" stroke-opacity="{opacity:.2f}" stroke-width="{1 + strength * 2:.1f}"/>'
        )

    # Draw outer nodes
    for i, node in enumerate(nodes):
        angle = (2 * math.pi * i) / len(nodes) - math.pi / 2
        nx = cx + radius * math.cos(angle)
        ny = cy + radius * math.sin(angle)
        co = node.get("co_occurrence_count", 1)
        node_r = 6 + (co / max_co) * 10
        name = node["canonical_name"]
        # Truncate long names
        display = name[:18] + "..." if len(name) > 18 else name
        eid = node.get("id", "")

        lines.append(f'<a href="/explore/entities/{eid}" class="graph-node">')
        lines.append(f'  <circle cx="{nx:.1f}" cy="{ny:.1f}" r="{node_r:.1f}" fill="var(--nx-accent-soft)" stroke="var(--nx-accent)" stroke-width="1.5"/>')
        # Position text based on angle
        text_anchor = "start" if math.cos(angle) > 0.1 else ("end" if math.cos(angle) < -0.1 else "middle")
        tx = nx + (node_r + 6) * (1 if math.cos(angle) > 0.1 else (-1 if math.cos(angle) < -0.1 else 0))
        ty = ny + (4 if abs(math.cos(angle)) > 0.1 else (node_r + 14 if math.sin(angle) > 0 else -node_r - 6))
        lines.append(f'  <text x="{tx:.1f}" y="{ty:.1f}" text-anchor="{text_anchor}">{display}</text>')
        lines.append('</a>')

    # Draw center node
    center_display = center_name[:22] + "..." if len(center_name) > 22 else center_name
    lines.append(f'<g class="graph-center">')
    lines.append(f'  <circle cx="{cx}" cy="{cy}" r="20" fill="var(--nx-accent)" stroke="var(--nx-bg)" stroke-width="3"/>')
    lines.append(f'  <text x="{cx}" y="{cy + 34}" text-anchor="middle" fill="var(--nx-text)">{center_display}</text>')
    lines.append('</g>')

    lines.append('</svg>')
    return '\n'.join(lines)
