"""
Rift — CLI entry point.

Launches the Streamlit application via subprocess so that users can
simply run ``python main.py`` from the project root.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Launch the Streamlit UI."""
    app_path = Path(__file__).resolve().parent / "ui" / "streamlit_app.py"

    if not app_path.exists():
        print(f"ERROR: Streamlit app not found at {app_path}", file=sys.stderr)
        sys.exit(1)

    print("Starting Rift...")
    print(f"   App: {app_path}")
    print("   Press Ctrl+C to stop.\n")

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                "--server.headless=true",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        print("\nRift stopped.")
    except FileNotFoundError:
        print(
            "ERROR: Streamlit is not installed. "
            "Run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: Streamlit exited with code {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()
