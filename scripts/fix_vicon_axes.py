"""Reverse IMU x and y in a Vicon dataset where they oppose the optical body.

Each CSV is checked against the optical angular rate.  Recordings that need it
have the accelerometer and gyroscope x and y columns sign-flipped in place.
Position, quaternion and the z axis are left as recorded.  Running the script
again is a no-op: a file that already agrees is not rewritten.

    python scripts/fix_vicon_axes.py data/imu_vicon_joint_v10
    python scripts/fix_vicon_axes.py data/imu_vicon_joint_v10_short --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from wattitude.data.vicon import correct_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", type=Path, help="dataset directory containing the CSV recordings")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report which files would change, without writing them",
    )
    args = ap.parse_args()

    results = correct_dataset(args.folder, dry_run=args.dry_run)
    n_fixed = 0
    for path, fix in results:
        if fix == "rz180":
            n_fixed += 1
            verb = "would reverse" if args.dry_run else "reversed"
            print(f"{path.name}: {verb} x and y")
        else:
            print(f"{path.name}: unchanged")
    print(f"{n_fixed} of {len(results)} files {'need' if args.dry_run else 'received'} the reversal")


if __name__ == "__main__":
    main()
