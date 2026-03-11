"""Interactive setup wizard — generates config.yaml and .env from user choices."""

from pathlib import Path

import yaml

PRESET_INFO = [
    ("free", "Free (Ollama local)", "No API key needed — runs models locally", None),
    ("cheap", "Cheap (DeepSeek)", "~$0.01/day — needs DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY"),
    ("balanced", "Balanced (Gemini)", "~$0.05/day — needs GEMINI_API_KEY", "GEMINI_API_KEY"),
    ("quality", "Quality (Gemini Pro)", "~$0.15/day — needs GEMINI_API_KEY", "GEMINI_API_KEY"),
]

TOPIC_CHOICES = [
    ("iran-us-relations", "Iran-US Relations"),
    ("ai-ml-research", "AI/ML Research"),
    ("formula-1", "Formula 1"),
    ("global-energy-transition", "Global Energy Transition"),
]


def run_setup(data_dir: Path) -> None:
    """Interactive setup: pick preset, paste key, pick topics, generate config."""
    print("\n=== Nexus Setup Wizard ===\n")

    # 1. Pick preset
    print("Choose a model preset:\n")
    for i, (_, label, desc, _) in enumerate(PRESET_INFO, 1):
        print(f"  {i}. {label} — {desc}")
    choice = int(input("\nPreset [1-4]: ").strip())
    preset_name, _, _, required_key = PRESET_INFO[choice - 1]

    # 2. API key (if needed)
    api_key = None
    if required_key:
        api_key = input(f"\nPaste your {required_key}: ").strip()

    # 3. Pick topics
    print("\nAvailable topics:\n")
    for i, (_, name) in enumerate(TOPIC_CHOICES, 1):
        print(f"  {i}. {name}")
    picks = input("\nSelect topics (comma-separated numbers, e.g. 1,3): ").strip()
    selected = []
    for p in picks.split(","):
        idx = int(p.strip()) - 1
        if 0 <= idx < len(TOPIC_CHOICES):
            slug, name = TOPIC_CHOICES[idx]
            selected.append({"name": name, "priority": "high"})

    if not selected:
        selected = [{"name": TOPIC_CHOICES[0][1], "priority": "high"}]

    # 4. User info
    user_name = input("\nYour name: ").strip() or "User"
    timezone = input("Timezone (e.g. America/New_York) [UTC]: ").strip() or "UTC"

    # 5. Generate config.yaml
    config = {
        "preset": preset_name,
        "user": {"name": user_name, "timezone": timezone, "output_language": "en"},
        "topics": selected,
        "briefing": {"schedule": "06:00"},
        "audio": {"enabled": preset_name != "free"},
        "breaking_news": {"enabled": True, "threshold": 7},
        "telegram": {"enabled": True},
    }

    data_dir.mkdir(parents=True, exist_ok=True)
    config_path = data_dir / "config.yaml"
    config_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))
    print(f"\nWrote {config_path}")

    # 6. Generate .env
    env_path = data_dir.parent / ".env"
    env_lines = []
    if api_key and required_key:
        env_lines.append(f"{required_key}={api_key}")
    if env_lines:
        env_path.write_text("\n".join(env_lines) + "\n")
        print(f"Wrote {env_path}")

    print(f"\nDone! Run: python -m nexus run")
    print(f"Dashboard: http://localhost:8080")
