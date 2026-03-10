# System Map

Single-page static microsite for understanding the current Nexus architecture from code, with planned components from `design.md` shown as roadmap overlays.

## Files

- `index.html`: only entrypoint
- `styles.css`: visual system and layout
- `app.js`: JSON-driven rendering and interactions
- `data/graph.json`: semantic graph, data model, storage map, and source references

## Open locally

Serve the repo root or the `docs/system-map/` directory with any static server. Example from the repo root:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/docs/system-map/
```

## Maintenance

- Treat `data/graph.json` as the canonical content source.
- Update nodes, edges, pipeline stages, data models, storage entries, and source refs there first.
- Keep roadmap elements explicitly marked with `"status": "planned"`.
- Keep anything only partially wired through the runtime marked `"partial"`.

## Page behavior

- `Implemented only` hides planned items but keeps partial current-code elements visible.
- `Include roadmap` toggles planned components and planned edges.
- `Show storage` toggles filesystem-backed artifact views.
- `Show LLM boundaries` toggles model-backed components from the rendered views.
