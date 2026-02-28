"""Launch both the FastAPI backend and React frontend for development."""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent


def main():
    processes = []

    try:
        # Start FastAPI backend
        print("\n  🚀 Starting FastAPI backend on http://localhost:8000")
        backend = subprocess.Popen(
            [
                "uv",
                "run",
                "uvicorn",
                "api.main:app",
                "--reload",
                "--port",
                "8000",
            ],
            cwd=str(ROOT),
            shell=True,
        )
        processes.append(backend)

        # Install frontend deps if needed
        frontend_dir = ROOT / "frontend"
        if not (frontend_dir / "node_modules").exists():
            print("  📦 Installing frontend dependencies...")
            subprocess.run(
                ["npm", "install"],
                cwd=str(frontend_dir),
                check=True,
                shell=True,
            )

        # Start React frontend
        print("  ⚛️  Starting React frontend on http://localhost:5173")
        frontend = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=str(frontend_dir),
            shell=True,
        )
        processes.append(frontend)

        print("\n  ✅ Both servers running!")
        print("     Backend API:  http://localhost:8000/api/health")
        print("     Frontend UI:  http://localhost:5173")
        print("     Press Ctrl+C to stop both.\n")

        # Wait for either process to exit
        while True:
            for p in processes:
                if p.poll() is not None:
                    raise KeyboardInterrupt
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n  🛑 Shutting down...")
        for p in processes:
            p.terminate()
        for p in processes:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        print("  Done.\n")


if __name__ == "__main__":
    main()
