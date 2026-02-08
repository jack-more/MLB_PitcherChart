"""
Pipeline orchestrator: runs all steps in sequence.

Usage:
    python pipeline/run_all.py           # Run all steps
    python pipeline/run_all.py --from 3  # Resume from step 3
"""

import subprocess
import sys
import os
import argparse
import time

STEPS = [
    ("01_fetch_statcast.py", "Fetch Statcast data"),
    ("02_fetch_pitcher_roles.py", "Classify SP/RP roles"),
    ("03_feature_engineering.py", "Engineer pitcher-season features"),
    ("04_clustering.py", "K-Means clustering"),
    ("05_cluster_naming.py", "Generate archetype names"),
    ("06_hitter_vs_cluster.py", "Compute hitter vs cluster stats"),
]

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="Run the MLB Pitcher Archetype pipeline")
    parser.add_argument(
        "--from", dest="start_step", type=int, default=1,
        help="Step number to start from (1-6). Default: 1",
    )
    args = parser.parse_args()

    start_idx = args.start_step - 1
    if start_idx < 0 or start_idx >= len(STEPS):
        print(f"Invalid step number. Must be 1-{len(STEPS)}")
        sys.exit(1)

    steps_to_run = STEPS[start_idx:]
    print(f"Running {len(steps_to_run)} pipeline steps:")
    for i, (script, desc) in enumerate(steps_to_run, start=start_idx + 1):
        print(f"  {i}. {desc} ({script})")
    print()

    for i, (script, desc) in enumerate(steps_to_run, start=start_idx + 1):
        script_path = os.path.join(PIPELINE_DIR, script)
        print(f"\n{'='*60}")
        print(f"Step {i}/{len(STEPS)}: {desc}")
        print(f"Script: {script}")
        print(f"{'='*60}\n")

        start_time = time.time()

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                check=True,
                cwd=os.path.dirname(PIPELINE_DIR),  # project root
            )
        except subprocess.CalledProcessError as e:
            print(f"\nFAILED at step {i}: {script}")
            print(f"Exit code: {e.returncode}")
            print(f"Resume with: python pipeline/run_all.py --from {i}")
            sys.exit(e.returncode)

        elapsed = time.time() - start_time
        print(f"\nCompleted step {i} in {elapsed:.1f}s")

    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
