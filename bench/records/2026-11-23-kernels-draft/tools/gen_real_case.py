"""Generate the drafter oracle planes of one real target fixture (a single-case driver over oracle/gen_dspark_fixtures.py).

    gen_real_case.py --pack PACK --fixtures DIR --out DIR --name FIXTURE --anchors 1,5,13,...
The case is named real_FIXTURE; its features are stack(hidden[5], hidden[23], hidden[41], h_final) of FIXTURE.safetensors."""
import argparse, sys
sys.path.insert(0, "oracle")
import gen_dspark_fixtures as g

ap = argparse.ArgumentParser()
ap.add_argument("--pack", required=True)
ap.add_argument("--fixtures", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--anchors", required=True)
a = ap.parse_args()
case = "real_" + a.name
g.CASES[case] = dict(fixture=a.name, anchors=[int(x) for x in a.anchors.split(",")])
sys.exit(g.main(["--pack", a.pack, "--fixtures", a.fixtures, "--out", a.out, "--threads", "8", "--only", case]))
