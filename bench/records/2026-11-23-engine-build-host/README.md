# Engine build and host tests on node-2 (plan step 4.8 precursor, gate `engine-build`)

Lab build of engine commit `df4862be52cb831072ce420d81412f0eae7039a8` (integration head: upstream main, the Small family and the Mini
host half) through the controller, then `engine-test host` with CUDA enabled and no server running.

- `build.txt`: controller output (rexl3-serve sha256, 2m17.8s wall). `version.txt`: `rexl3-serve --version` (commit, sm_121).
- `engine-test-host.txt`: four suites, 64 + 19 + 1 + 18 tests passed, none failed. The 64 gb-model tests are the 31 Small
  tests plus the Mini host-half tests.
