"""CLI entry point: python -m nexus <command>."""

import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m nexus <engine|setup>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "engine":
        print("Engine pipeline not yet implemented.")
    elif command == "setup":
        print("Setup wizard not yet implemented.")
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
