import sys
from pathlib import Path

# Add the project root to sys.path so `import app` works when running
# pytest from any working directory (required because pyproject.toml
# sets [tool.uv] package = false — the project is an app, not a library).
sys.path.insert(0, str(Path(__file__).resolve().parent))
