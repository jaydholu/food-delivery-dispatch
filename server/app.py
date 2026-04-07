"""
FastAPI application for the Food Delivery Dispatch OpenEnv Environment.

Endpoints (OpenEnv spec):
    POST /reset      → FoodDeliveryObservation (initial state)
    POST /step       → FoodDeliveryObservation (next state)
    GET  /state      → episode_id + step_count
    GET  /schema     → action / observation schemas
    WS   /ws         → WebSocket persistent session
    GET  /web        → Interactive Web UI
    GET  /health     → { status: "ok" }

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

import os

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv-core is required. Install with: pip install openenv-core"
    ) from e

try:
    from ..models import FoodDeliveryAction, FoodDeliveryObservation
    from .food_delivery_openenv_environment import FoodDeliveryEnvironment
    
except (ImportError, ModuleNotFoundError):
    from models import FoodDeliveryAction, FoodDeliveryObservation
    from server.food_delivery_openenv_environment import FoodDeliveryEnvironment


# Task is configurable via environment variable (default: medium)
_TASK = os.getenv("FOOD_DELIVERY_TASK", "medium")


def _make_env() -> FoodDeliveryEnvironment:
    """Factory that creates a fresh environment instance per session."""
    return FoodDeliveryEnvironment(task=_TASK)


app = create_app(
    _make_env,                  # factory mode — each WebSocket session gets its own env
    FoodDeliveryAction,
    FoodDeliveryObservation,
    env_name="food_delivery_openenv",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task", type=str, default="medium", choices=["easy", "medium", "hard"])
    args = parser.parse_args()

    os.environ["FOOD_DELIVERY_TASK"] = args.task

    main(port=args.port)
