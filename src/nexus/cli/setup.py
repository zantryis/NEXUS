"""Interactive setup wizard — generates config.yaml and .env from user choices."""

from pathlib import Path

from nexus.config.writer import write_config, write_env

PRESET_INFO = [
    ("free", "Free (Ollama local)", "No API key needed — runs models locally", None),
    ("cheap", "Cheap (DeepSeek)", "~$0.01/day — needs DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY"),
    ("balanced", "Balanced (Gemini)", "~$0.05/day — needs GEMINI_API_KEY", "GEMINI_API_KEY"),
    ("quality", "Quality (Gemini Pro)", "~$0.15/day — needs GEMINI_API_KEY", "GEMINI_API_KEY"),
    ("openai-cheap", "OpenAI Cheap", "~$0.03/day — needs OPENAI_API_KEY", "OPENAI_API_KEY"),
    ("openai-balanced", "OpenAI Balanced", "~$0.10/day — needs OPENAI_API_KEY", "OPENAI_API_KEY"),
    ("anthropic", "Anthropic (Claude)", "~$0.10/day — needs ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
]

TOPIC_CHOICES = [
    ("iran-us-relations", "Iran-US Relations"),
    ("ai-ml-research", "AI/ML Research"),
    ("formula-1", "Formula 1"),
    ("global-energy-transition", "Global Energy Transition"),
]


def _prompt_preset() -> int:
    """Prompt for preset selection with retry on invalid input."""
    print("Choose a model preset:\n")
    for i, (_, label, desc, _) in enumerate(PRESET_INFO, 1):
        print(f"  {i}. {label} — {desc}")
    while True:
        try:
            choice = int(input(f"\nPreset [1-{len(PRESET_INFO)}]: ").strip())
            if 1 <= choice <= len(PRESET_INFO):
                return choice
            print(f"  Please enter a number between 1 and {len(PRESET_INFO)}.")
        except ValueError:
            print(f"  Please enter a number between 1 and {len(PRESET_INFO)}.")


def _prompt_api_key(required_key: str) -> str:
    """Prompt for API key with retry on empty input."""
    while True:
        key = input(f"\nPaste your {required_key}: ").strip()
        if key:
            return key
        print("  API key cannot be empty.")


def _prompt_topics() -> list[dict]:
    """Prompt for topic selection + optional custom topic."""
    print("\nAvailable topics:\n")
    for i, (_, name) in enumerate(TOPIC_CHOICES, 1):
        print(f"  {i}. {name}")
    picks = input("\nSelect topics (comma-separated numbers, e.g. 1,3): ").strip()
    selected = []
    for p in picks.split(","):
        try:
            idx = int(p.strip()) - 1
            if 0 <= idx < len(TOPIC_CHOICES):
                slug, name = TOPIC_CHOICES[idx]
                selected.append({"name": name, "priority": "high"})
        except ValueError:
            continue  # skip invalid entries

    # Custom topic option
    custom = input("\nAdd a custom topic? (name or Enter to skip): ").strip()
    if custom:
        selected.append({"name": custom, "priority": "medium"})

    if not selected:
        selected = [{"name": TOPIC_CHOICES[0][1], "priority": "high"}]

    return selected


def run_setup(data_dir: Path) -> None:
    """Interactive setup: pick preset, paste key, pick topics, generate config."""
    print("\n=== Nexus Setup Wizard ===\n")

    # 1. Pick preset
    choice = _prompt_preset()
    preset_name, _, _, required_key = PRESET_INFO[choice - 1]

    # 2. API key (if needed)
    api_key = None
    if required_key:
        api_key = _prompt_api_key(required_key)

    # 3. Pick topics
    selected = _prompt_topics()

    # 4. User info
    user_name = input("\nYour name: ").strip() or "User"
    timezone = input("Timezone (e.g. America/New_York) [UTC]: ").strip() or "UTC"

    # 5. Generate config.yaml
    config_dict = {
        "preset": preset_name,
        "user": {"name": user_name, "timezone": timezone, "output_language": "en"},
        "topics": selected,
        "briefing": {"schedule": "06:00"},
        "audio": {"enabled": preset_name != "free"},
        "breaking_news": {"enabled": True, "threshold": 7},
        "telegram": {"enabled": True},
    }

    config_path = write_config(data_dir, config_dict)
    print(f"\nWrote {config_path}")

    # 6. Generate / update .env
    env_keys = {}
    if api_key and required_key:
        env_keys[required_key] = api_key

    if env_keys:
        env_path = write_env(data_dir.parent, env_keys)
        print(f"Wrote {env_path}")

    print(f"\nDone! Run: python -m nexus run")
    print(f"Dashboard: http://localhost:8080")
