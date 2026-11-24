"""The committed record of tools/verify_exl3_real.py (real tensors of the pinned Inkling-2-Mini pack fetched by HTTP range,
compared with gb-exl3 bit for bit) must show every group matched and cover the required K classes."""

import json
import os

ORACLE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REVISION = "44c78c9d229b43dd45e41506071b9f5f939bed44"


def record():
    with open(os.path.join(ORACLE, "records", "exl3-real-verify.json")) as f:
        return json.load(f)


def test_record_header_and_every_entry_matched():
    r = record()
    assert r["repo"] == "vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw" and r["revision"] == REVISION
    assert r["entries"] and r["summary"]["all_equal"] is True
    for e in r["entries"]:
        assert e["match"] is True and e["wq_bits_equal"] is True and e["dense_bits_equal"] is True, e["tensor"]
        assert e["strips_equal_rust"] is True and e["decode_cols_equal"] is True and e["marker"] == "0x83dcd12d", e["tensor"]
        assert e["bytes"] > 0 and e["k"] == e["k_bits"] and e["source"]["file"].endswith(".safetensors")
    assert len({e["tensor"] for e in r["entries"]}) == len(r["entries"])
    assert set(r["groups"]) == {e["tensor"] for e in r["entries"]}


def test_record_covers_required_classes():
    es = record()["entries"]
    experts = [e for e in es if ".mlp.experts." in e["tensor"]]
    k4 = [e for e in experts if e["k"] == 4]
    k5 = [e for e in experts if e["k"] == 5]
    assert len(k4) >= 3 and len(k5) >= 2
    layers = {int(e["tensor"].split(".")[3]) for e in experts}
    assert any(6 <= x <= 17 for x in layers) and any(not 6 <= x <= 17 for x in layers)
    assert len({tuple(e["tensor"].split(".")[3:7]) for e in experts}) >= 4  # (layer, mlp, experts, id)
    attn = [e for e in es if ".attn." in e["tensor"] and e["k"] == 6]
    slabs = [e for e in es if e["tensor"].startswith("model.llm.unembed[") and e["k"] == 6]
    drafter = [e for e in es if e["tensor"].startswith("mtp:") and e["k"] == 4]
    vision = [e for e in es if e["tensor"].startswith("model.visual.") and e["k"] == 6]
    assert len(attn) >= 1 and len(slabs) >= 2 and len(drafter) >= 1 and len(vision) >= 1
    assert any(e["source"]["columns"][1] == 201088 for e in slabs) and any(e["source"]["columns"][0] == 0 for e in slabs)
    for e in slabs:
        assert e["source"]["file"] == "model-00002-of-00002.safetensors" and e["source"]["requests"] == 144


def test_record_summary_consistent():
    r = record()
    per_k = {}
    for e in r["entries"]:
        per_k[str(e["k"])] = per_k.get(str(e["k"]), 0) + 1
    s = r["summary"]
    assert s["entries"] == len(r["entries"]) and s["entries_per_k"] == per_k and s["all_markers_mul1"] is True
    assert 0 < s["total_bytes_fetched"] < 1 << 30  # ranges only, never a whole shard
    assert all(x["finite"] for x in r["experts_ffn_sanity"].values())
