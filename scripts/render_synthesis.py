"""Render synthesis YAML files into human-readable markdown. No LLM needed."""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from nexus.engine.synthesis.knowledge import TopicSynthesis


def render_topic(syn: TopicSynthesis) -> str:
    lines = [f"# {syn.topic_name}\n"]

    bal = syn.source_balance or {}
    total = sum(bal.values())
    bal_str = ", ".join(f"{k}: {v}" for k, v in sorted(bal.items(), key=lambda x: -x[1]))
    lines.append(f"**Sources:** {total} articles ({bal_str})")
    if syn.languages_represented:
        lines.append(f"**Languages:** {', '.join(syn.languages_represented)}")
    lines.append("")

    if syn.background:
        lines.append("## Background Context")
        for b in syn.background[-3:]:
            lines.append(f"- *{b.period_start} to {b.period_end}:* {b.text}")
        lines.append("")

    for i, t in enumerate(syn.threads, 1):
        lines.append(f"## Thread {i}: {t.headline}")
        lines.append(f"*Significance: {t.significance}/10*\n")

        if t.events:
            lines.append("### Events")
            for e in t.events:
                outlets = []
                for s in e.sources:
                    aff = s.get("affiliation", "?")
                    country = s.get("country", "?")
                    outlet = s.get("outlet", "?")
                    outlets.append(f"{outlet} ({aff}/{country})")
                lines.append(f"- **[{e.date}]** {e.summary}")
                lines.append(f"  - Sources: {', '.join(outlets)}")
                lines.append(f"  - Entities: {', '.join(e.entities)}")
                if hasattr(e, "relation_to_prior") and e.relation_to_prior:
                    lines.append(f"  - *Context: {e.relation_to_prior}*")
            lines.append("")

        if t.convergence:
            lines.append("### Convergence (agreed facts)")
            for c in t.convergence:
                if isinstance(c, dict):
                    confirmed = ", ".join(c.get("confirmed_by", []))
                    lines.append(f"- {c.get('fact', '?')} — confirmed by: **{confirmed}**")
                else:
                    lines.append(f"- {c}")
            lines.append("")

        if t.divergence:
            lines.append("### Divergence (framing differences)")
            for d in t.divergence:
                shared = d.get("shared_event", d.get("claim", "?"))
                lines.append(f"- **{shared}**")
                lines.append(f"  - {d.get('source_a', '?')}: \"{d.get('framing_a', '?')}\"")
                lines.append(f"  - {d.get('source_b', '?')}: \"{d.get('framing_b', '?')}\"")
            lines.append("")

        if t.key_entities:
            lines.append(f"*Key entities: {', '.join(t.key_entities)}*\n")

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Render synthesis YAML to markdown")
    parser.add_argument("path", nargs="?", help="Path to synthesis YAML or directory of YAMLs")
    parser.add_argument("--provider", default="gemini", help="Provider label (default: gemini)")
    parser.add_argument("--date", default="2026-03-09", help="Date (default: 2026-03-09)")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    args = parser.parse_args()

    if args.path:
        paths = [Path(args.path)] if Path(args.path).is_file() else sorted(Path(args.path).glob("*.yaml"))
    else:
        base = Path("data/backtest") / args.provider / "artifacts" / "syntheses" / args.date
        paths = sorted(base.glob("*.yaml"))

    if not paths:
        print(f"No synthesis files found", file=sys.stderr)
        sys.exit(1)

    sections = []
    for p in paths:
        raw = yaml.safe_load(p.read_text())
        syn = TopicSynthesis(**raw)
        sections.append(render_topic(syn))

    output = f"---\n\n".join(sections)
    header = f"# Daily Intelligence Briefing — {args.date} ({args.provider.title()})\n\n---\n\n"
    output = header + output

    if args.output:
        Path(args.output).write_text(output)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
