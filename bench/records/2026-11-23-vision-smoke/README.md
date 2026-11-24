# vision-smoke: images through the Mini's server on one Spark (gate `vision-smoke`)

Result **pass** (`meta.json`). Node-2, lab build of the committed engine head `bfe68ae1a087cd3b6976db711675151c410d14a8` (branch
`inkling2-loop-backend`, `dirty` false), the `bootstrap` profile with `VISION=on`, `SPEC=none`, `PREFIX_CACHE=off`,
`REXL3_ALLOW_UNGATED_COMPONENTS=1` (short_conv and forward still ungated, named at boot). The tower is `vision_inkling2_loop`
(`kernels-vision` record), now in `BUILT_COMPONENTS` together with its serving path. Requests: `vision_smoke.py` (stdlib; images from
`bench/tools/synth_images.py`: a red square, a blue circle, a green/yellow split, 240x240 PNGs as data URLs), greedy, `return_token_ids`.
Raw request/reply pairs (image data elided) are in `replies/`, the run's printout in `run.txt`.

| Check | Result |
|---|---|
| each image alone names its colours (and shape) | pass: "a red square/rectangle on a white background", "a blue circle", "the left half is green and the right half is yellow"; 70, 70 and 72 prompt tokens, 1.6 to 1.8 s each |
| the same request again gives the same token ids | pass for all three (49, 58 and 51 generated ids identical) |
| three images in one request | pass: "image1: red square. image2: blue circle. image3: split green/yellow", 145 prompt tokens, 3.1 s |
| two image requests and a text request at once | pass: each generated-id list equals its own solo run (red, blue, text) |
| an image of exactly the cap (64 x 64 cells = 4,096 tokens, 2,560 px square of red and white stripes) | pass: HTTP 200 in 8.3 s, 4,121 prompt tokens (the prefill takes two 4,096-row chunks, so the splice crosses a chunk boundary); the reply describes "horizontal alternating red and white stripes" |
| an image of 68 x 68 = 4,624 tokens | refused by name, HTTP 400: `unsupported image: 2720x2720 is 68 x 68 = 4624 tokens of 40 pixels, over the cap of 4096 (Inkling 2 does not resize)` |
| `--no-vision` path unchanged | `VISION=off` boot: capacity line equals the text-only boot's (`tower_gib=0.00 vision_reserve_gib=0.00`), and the four text replies of `smoke_curl.py` equal the first-boot smoke's (`Paris`; 17 x 23 = 391; the colours completion; `get_weather({"city":"Paris"})`) |

## Memory (`serve-and-mem.txt`, `status-ready.txt`, `memwatch.txt`)

* Plan with the tower loaded: `tower_gib=0.79 vision_reserve_gib=0.28`, `free_after_gib=90.66`, KV pool unchanged (4,100 pages, 16.14 GiB).
* The loaded tower's measured scratch at the 4,096-token cap: `scratch_bytes=596142492` (568.5 MiB, 16,385 rows), against the 541 MiB the
  plan priced before this change and the 577 MiB linear extrapolation of `kernels-vision`'s 72.1 MiB at 2,049 rows. The workspace part is
  not linear (8.7 KB a row at 2,049 rows, 7.8 KB at 16,385), so `memory::TOWER_WORKSPACE_BYTES_PER_ROW` (8,800) covers both points and
  the host test pins both (the model reads 72.1 MiB to +3 % at 512 tokens and 568.5 MiB to +3 % at the cap).
* The server process holds 40,443 MiB (nvidia-smi) with the tower, 39,617 MiB text-only on node-4 (`first-boot`): +826 MiB for the weights
  (0.226 GiB), the scratch and the encoder's stream, against 0.791 GiB (810 MiB) planned. Not the same node, so the 16 MiB over is not a
  clean difference.
* MemAvailable minimum: 66,703 MiB while loading, 67,122 MiB once ready (node-2 carries the fixtures job; its idle baseline is lower than
  node-4's: planned free after weights 90.7 against 101.5 GiB).

## Not proven

* The image replies are judged by eye on synthetic pictures; no comparison against an image-answering reference, no real photo, no text
  in an image, no video (the family has none).
* A batch of images is not compared byte for byte against each image alone at the server: the encoder takes one image at a time, the
  tower's batch == alone is `kernels-vision`'s (bit for bit, four real images). The three-image request and the solo repeats are what
  the server shows.
* The short convolution and the forward are still ungated; replies are unqualified until the oracle comparison. One slot of the
  four was loaded at a time except in the concurrent check (three requests); no run with four images in flight.
* Images in an engine with a prefix cache or a drafter: both are refused for this family.
