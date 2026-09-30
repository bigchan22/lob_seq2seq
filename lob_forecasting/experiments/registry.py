from pathlib import Path
def registry_path(): return Path(__file__).resolve().parents[2]/"experiments/registry.yaml"
