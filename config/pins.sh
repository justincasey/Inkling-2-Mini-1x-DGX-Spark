# Pins for Inkling-2-Mini-1x-DGX-Spark. Sourced by the controller; not tunable from .env.
MODEL_REPO=vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw
MODEL_REV=44c78c9d229b43dd45e41506071b9f5f939bed44
MODEL_NAME=Inkling-2-Mini-SAGE-REXL3-4.23bpw
MANIFEST_FILE=intake/manifest.tsv
MANIFEST_SHA256=ebaac398d3e25a2c6756304ce42cfaea3d5222fa46a383f581f633447b2f0875   # sha256 of MANIFEST_FILE: the byte authority for the pack

# Engine: REXL3 fork branch `inkling2-loop`, started 2026-11-22 from upstream main 02ef8cb (engine source tree v0.1.1,
# `.rexl3_commit` 75031112). REXL3_COMMIT is the engine pin: the commit the shipped build and the pinned references
# (`bench pin`: bench/reference.json, reference-vision.json) come from. It was empty while the family was brought up and is
# set to the integrated head after the stretch lanes (2026-11-23); a later lane merge moves it and `bench pin` then has to
# reproduce the references byte for byte. `build REV` is a lab build of any commit in the local engine repository that
# descends from REXL3_UPSTREAM_BASE.
REXL3_UPSTREAM=https://github.com/REXL3/REXL3.git
REXL3_UPSTREAM_BASE=02ef8cba920e1a86b911c6e20c248c43724ddd66
REXL3_FORK=https://github.com/justincasey/REXL3.git
# PUBLIC PIN (this copy of the repository): the head of the clean 46-topic, 185-commit series on the fork, branch `inkling2-mini`.
# Every record in bench/records/ was measured at the private development head 59d08a9c5e9f9d3ce96ed9d2353c68b70169cc1a, whose tree
# equals this commit's except for 25 weight-derived test files that cannot be redistributed (README, "The engine pin in this copy").
REXL3_COMMIT=fb5e3a7bbe590db3dc7ac039dd44fe751b92e3fc
RUST_TOOLCHAIN=1.89.0

# The GPU SM clock cap, MHz. Hardware protection, non-negotiable (owner, 2026-11-23): in place whenever any work runs on a node. The controller
# and scripts/ops_dev.sh apply it (sudo -n nvidia-smi -lgc CAP,CAP) before every server start and every GPU test and require the success line;
# nothing releases, raises or changes it, and it is not a .env setting (GPU_CLOCKS must be absent or equal to this).
GPU_CLOCK_CAP_MHZ=2200
