#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# gpu_cap.sh NODE: apply the GPU SM clock cap on NODE and require its success line, or fail.
#
# The cap (GPU_CLOCK_CAP_MHZ in config/pins.sh: 2200 MHz) is hardware protection and non-negotiable (owner, 2026-11-23): it is in place
# whenever any work runs on a node. Every script that runs GPU work on a node calls this first (tests/test_recipe.sh scans for it);
# the controller (inkling2mini.sh) and scripts/ops_dev.sh apply it the same way. `nvidia-smi -lgc` does not survive a reboot or a driver
# reload, and an idle clock reading near the cap does not prove it, so it is applied, not assumed; applying the same value again is
# harmless. There is no release, no other value and no skip: nothing in this recipe lowers or removes the cap.
#
# Environment: SSH_USER (spark). Exit 0 only when the node answered `GPU clocks set to "(gpuClkMin CAP, gpuClkMax CAP)"`.
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
. "$ROOT/config/pins.sh"

die() { echo "gpu_cap: $*" >&2; exit 1; }
NODE=${1:-}
[ -n "$NODE" ] || { echo "usage: gpu_cap.sh NODE" >&2; exit 2; }
case ${GPU_CLOCK_CAP_MHZ:-} in "" | *[!0-9]* | 0) die "GPU_CLOCK_CAP_MHZ in config/pins.sh must be a whole number above 0" ;; esac
SSH_USER=${SSH_USER:-spark}

out=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$SSH_USER@$NODE" "sudo -n nvidia-smi -lgc $GPU_CLOCK_CAP_MHZ,$GPU_CLOCK_CAP_MHZ 2>&1" </dev/null) \
  && grep -qF "gpuClkMin $GPU_CLOCK_CAP_MHZ, gpuClkMax $GPU_CLOCK_CAP_MHZ" <<<"$out" \
  || die "the GPU cap of $GPU_CLOCK_CAP_MHZ MHz could not be applied on $NODE (sudo -n nvidia-smi -lgc $GPU_CLOCK_CAP_MHZ,$GPU_CLOCK_CAP_MHZ): no GPU work without it${out:+: $out}"
echo "GPU SM clock capped at $GPU_CLOCK_CAP_MHZ MHz on $NODE"
