"""Trigger the deployed uav-fog-taskdist app's driver function and exit.

Unlike ``modal run --detach``, which ties an *ephemeral* app's lifetime to
the local session in ways that did not survive this session being closed for
an extended period (training progress was preserved by modal_taskdist.py's
own resumable checkpoints, but the orchestrating App itself disappeared),
this triggers a function on a *deployed* app -- fully independent of any
local client or session from the moment it is spawned.

Usage:
    conda run -n venv modal deploy modal_taskdist.py   # once, or after any edit
    conda run -n venv python3 modal_taskdist_trigger.py --mode full
"""

import argparse

import modal

from modal_taskdist import PRESETS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="full", choices=sorted(PRESETS))
    parser.add_argument("--run-id", default="taskdist-final-broker")
    args = parser.parse_args()

    preset = PRESETS[args.mode]
    spec = {"run_id": f"{args.run_id}-{args.mode}", **preset}

    driver = modal.Function.from_name("uav-fog-taskdist", "driver")
    call = driver.spawn(spec)
    print(f"Spawned driver call {call.object_id} for {spec['run_id']} "
          "on the DEPLOYED app -- independent of this process or session.")
    print("Recover with:")
    print("  modal app list        # find the app")
    print(f"  modal volume ls uav-fog-taskdist-results {spec['run_id']}")


if __name__ == "__main__":
    main()
