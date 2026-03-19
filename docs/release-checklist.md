# Release Checklist

Pre-release sign-off for the local-first onboarding path and default CI gate.

## Automated Gate

Run these commands from the repo root:

```bash
ruff check src tests
pytest -m "not e2e and not integration"
pip-audit --strict --desc
docker compose config
docker compose -f docker-compose.dev.yml config
docker compose -f docker-compose.test.yml config
```

Expected result:

- `ruff` exits cleanly
- `pytest` passes without collecting `e2e` or `integration` tests
- `pip-audit` reports no known vulnerabilities
- Compose config renders with `./.env:/app/.env` and `NEXUS_TRUST_DOCKER_GATEWAY=1`

## Manual Audit: Local Python

Use a clean checkout or temporary copy.

1. Create a local env file: `cp .env.example .env`
2. Start the app: `python -m nexus run`
3. Open `http://localhost:8080`
4. Complete the setup wizard with a test API key and at least one topic
5. Confirm the wizard wrote both:
   - `data/config.yaml`
   - project-root `.env`
6. Open Settings, change the API key, and click `Restart Now`
7. Confirm the restarted app uses the updated key instead of the pre-restart value
8. Confirm `/setup` and `/settings` are reachable from the same machine without `NEXUS_ADMIN_TOKEN`
9. Confirm a non-loopback client requires `NEXUS_ADMIN_TOKEN`

## Manual Audit: Local Docker

Use the default Compose file on the same machine that will run the browser.

1. Create the host env file: `cp .env.example .env`
2. Start the stack: `docker compose up --build`
3. Open `http://localhost:8080`
4. Complete the setup wizard from the host browser
5. Confirm the wizard updated the host project `.env`
6. Confirm `/setup` and `/settings` work from the host browser without `NEXUS_ADMIN_TOKEN`
7. Stop and recreate the container: `docker compose down && docker compose up --build`
8. Confirm `data/config.yaml` and the host `.env` changes persist
9. Confirm the restarted container uses the updated API key

## Original Findings Closure

Before tagging, explicitly verify each item below as closed:

- Admin routes are loopback-only by default, with only the opt-in Docker gateway exception enabled in shipped Compose files
- Wizard-entered secrets persist in the project-root `.env` for both local Python and Docker flows
- In-app restart reloads `.env` with override semantics and does not keep stale process values
- Default CI/test documentation excludes both `e2e` and `integration` tests
- `pip-audit` is clean, including the prior `pyasn1` finding

## Forward Look Packaging Checks

Before tagging, verify the release-facing forecast surface matches the `v0.1` scope:

- `/forward-look` is the canonical public route and renders `Forward Look`
- `/predictions` redirects to `/forward-look`
- The public page only shows actor-engine forecasts even if other engine rows exist in storage
- Dashboard topic cards only show actor-engine Forward Look entries
- Topic detail pages only show actor-engine Forward Look entries
- Kalshi sidebar cards only show actor-engine market matches
- Actor freeform and thread forecasts render without Kalshi copy when no market metadata exists
- Actor Kalshi-aligned forecasts show market comparison only when Kalshi metadata exists
- `/benchmark` remains directly reachable but is absent from normal navigation
- `python -m nexus` help and README examples only advertise the kept release-facing forecast flow

## Docs And Setup Consistency Checks

Before tagging, verify the release docs and onboarding defaults tell one coherent story:

- `README.md`, `docs/index.html`, and `docs/pipeline.html` all describe actor-based Forward Look with optional Kalshi
- Public docs do not advertise `6 competing forecast engines` or route users to `/predictions`
- `docs/pipeline.html` is the canonical public system map, and `docs/system-map/` is treated as legacy/internal
- `data/config.example.yaml`, CLI setup, and web setup all agree on the default briefing schedule/style, Telegram enabled, source discovery enabled, and `filter_threshold: 4.0`

## Sign-Off

Record the release candidate commit SHA, the date of the audit, and who ran:

- Automated gate
- Local Python audit
- Local Docker audit
