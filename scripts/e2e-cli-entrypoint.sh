#!/bin/bash
# Entrypoint for E2E CLI setup containers.
# Pipes CLI_SETUP_INPUT into `nexus setup`, then starts the web server.
set -e

if [ -z "$CLI_SETUP_INPUT" ]; then
    echo "ERROR: CLI_SETUP_INPUT env var not set" >&2
    exit 1
fi

echo "=== Running CLI setup ==="
printf "$CLI_SETUP_INPUT" | python -m nexus setup

echo "=== Starting server ==="
exec python -m nexus run --host 0.0.0.0
