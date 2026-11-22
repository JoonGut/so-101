"""Ejecuta la API oficial de LeRobot en un proceso con entrada interactiva."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Calibración oficial del SO-101 follower")
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.utils.utils import init_logging

    init_logging()
    follower = SO101Follower(SO101FollowerConfig(
        port=args.port, id=args.id, calibration_dir=args.directory,
    ))
    try:
        follower.connect(calibrate=False)
        follower.calibrate()
    finally:
        if follower.is_connected:
            follower.disconnect()


if __name__ == "__main__":
    main()
