# Plan — Inkling-2-Mini-1x-DGX-Spark

Closed 2026-11-23. The complete plan, with every step, its status, the gates, the engine work breakdown, decisions, risks and the log, is
[docs/exec-plans/completed/inkling2-mini-rexl3.md](docs/exec-plans/completed/inkling2-mini-rexl3.md); step status lives only there, so this file does not drift.
The results are in the README (generated from `bench/records/`), the gates in `EVAL-MATRIX.md`, the open items in the tech-debt tracker.

## Lane

- pack: `vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw` @ `44c78c9d229b43dd45e41506071b9f5f939bed44`: 13 files, 14.00 GiB of weights, sha256 in `intake/manifest.tsv`.
  EXL3 mixed K read from the trellis shapes: experts 4/5-bit, attention / unembed / vision 6-bit, drafter 4-bit, embedding BF16 (`docs/ROOFLINE.md`).
- model: `Inkling2LoopForConditionalGeneration` (26.4B total, 3.4B active): 24 layers, the middle 12 run three times with shared weights = 48 layer passes a token (8 global-attention,
  40 sliding-window), 24 query and 8 KV heads, 128 experts with top 8, a vision tower, a block drafter (7 drafted tokens a round).
- topology: **TP1, one Spark** (owner, 2026-11-22). The latest build serves on node-1 for the owner to test; the Inkling-2-Small TP4 group, which
  needs all four Sparks, is stopped and not restored (`./inkling2mini.sh restore-other` brings it back; `docs/CLEANUP.md`).
- engine: **REXL3, new family `inkling2_loop`**, built from the `inkling2` family; local branch `inkling2-loop` (worktree `~/projects/rexl3-inkling2-loop`) on upstream `main`
  `02ef8cb`; the pin is `REXL3_COMMIT` in `config/pins.sh`. A clean `engine/<topic>` series of the same change: `docs/ENGINE_EXPORT.md`; notes for a submission: `docs/REXL-PR-NOTES.md`. On 2026-11-24, at the owner's instruction, the series head was pushed as the branch `inkling2-mini` of the owner's fork `justincasey/REXL3` and offered to REXL as a `[finding]` issue (its URL is in the README); the development branch `inkling2-loop` is not pushed and must never be (25 weight-derived test files).
- targets: the owner's three stretch targets, all byte-exact: cold prefill over 3,000 tok/s at 32K, mean decode over 100 tok/s (c=1, four kinds, drafter on), c=4 aggregate over 200 tok/s.
  **None was met** (README, `docs/PREFILL-SPEED.md`, `docs/DECODE-SPEED.md`, `docs/DECODE-MEAN4-PLAN.md`, `docs/C4-DECODE-PLAN.md`). Of the earlier proposals in `docs/ROOFLINE.md` (Targets): cold prefill at 32K (2,120) and the
  4-stream aggregate (89) were met; serial decode at 4K (62) was not (51.3, record `graph-decode`).

## Risks that were open at the start, and how they ended

- **No engine.** Built: the `inkling2_loop` family (loop passes, mixed K, attention at 24 query and 8 KV heads, loop controllers, 6-bit dense EXL3, drafter, vision tower, prefix cache, CUDA graphs, expert arena,
  memory plan) serves the pack; the Small family stays byte-exact (31 host tests, the Small's golden hashes on a node).
- **Hardware not measured.** Measured on a node: `bench/records/2026-11-23-roofline/` (read bandwidth and bf16 rate within 3% of the recorded figures).
- **MTP may not pay.** It pays on code, structured output, JSON and tool calls and loses on plain prose, so `SPEC=dspark` ships and a chat-only deployment should set `SPEC=none` (`docs/DECISIONS.md`).
- **Pack identity.** Unchanged: the card's base revision `6e7c8248...` is not served by the base repo (`docs/DECISIONS.md`); the oracle checks the engine against the pack itself.
- **Routing assumed uniform.** Measured: eight consecutive tokens touch 28-33 distinct experts a visit against 8 for one (`docs/C4-DECODE-PLAN.md`).
- **Capacity at the edge.** The profiles `bootstrap` (shipped), `long` and `max` are measured (`bench/records/2026-11-23-profile-*`); the memory plan is optimistic by about 1.35 GiB at `max` with the drafter and the vision tower (README limits, `docs/FIT.md`).
- **The base card's warnings** (text loops, destructive tool use): the tool gate saw no unrequested destructive call; raw text continuation loops on half of the probe prompts, reasoning mode on none
  (`bench/records/2026-11-23-final-quality-tools/`, whose tool gate is a recorded fail at 25 of 28 cases).

## Decisions

All decided in the plan's Decisions table (D1-D9; D3 and D4 are closed by the owner's stretch targets and the shipped `bootstrap` profile). Since then, at the owner's instruction (2026-11-24): this repository is published (public, `justincasey/Inkling-2-Mini-1x-DGX-Spark`, an audited single-commit export; the lab's git history is not published) and the series is offered to REXL (above). Still the owner's:

- Whether to restore the Small (the Mini holds node-1), and whether to delete the quarantined lane branch `lane/k-clock`, the lane worktrees and the lab trees (`docs/CLEANUP.md`; the owner chose to keep everything for now).
- How REXL wants the series delivered if they answer: rebased onto their current `main`, split into separate findings, which pieces first (`docs/REXL-PR-NOTES.md` section 6).

## Acceptance criteria

Boots on one Spark behind an OpenAI-compatible endpoint (`final-boot`, `final-smoke`); the Small recipe's gate set (smoke, oracle, drafted == serial, cached == uncached, concurrent == solo, chunk width, boot to boot, needles at the window:
`EVAL-MATRIX.md`, all pass at the pin except the tool-call gate); every number in the README has a record under `bench/records/` (the table is generated from them: `scripts/records_index.py`).
