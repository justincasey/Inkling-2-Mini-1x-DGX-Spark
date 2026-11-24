#!/usr/bin/env bash
# SixCat speed profile against the node's OpenAI endpoint, run from the control machine (cross-recipe comparable: the same
# pinned SixCat and the same flags RED-SNOW-5.3-Flash uses).
#
#   bench/sixcat.sh --url http://NODE:PORT/v1 --model NAME --out FILE [--venv DIR] [--max-seconds N]
#
#   --url          the server's OpenAI base URL (with /v1)
#   --model        the served model name (the id /v1/models reports)
#   --out          where SixCat writes its JSON result
#   --venv         the virtualenv holding SixCat (default: bench/runs/sixcat-venv, git-ignored)
#   --max-seconds  SixCat's time budget (default: SIXCAT_MAX_SECONDS or 1200)
#
# The first run creates the venv and installs SixCat from the pin below; the install is skipped while the venv's `.commit`
# file equals the pin. Override the pin with SIXCAT_REPO / SIXCAT_COMMIT in the environment. Nothing is installed or
# downloaded by any other command. The engine has no API key: SixCat gets the literal `none`.
set -euo pipefail

SIXCAT_REPO="${SIXCAT_REPO:-https://github.com/vcruz305/sixcat-eval.git}"
SIXCAT_COMMIT="${SIXCAT_COMMIT:-841f90a4843745675c771ac4b5ae9205049b42c0}"   # v0.7.0
SIXCAT_MAX_SECONDS="${SIXCAT_MAX_SECONDS:-1200}"

die() { echo "sixcat.sh: $*" >&2; exit 1; }

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
url="" model="" out="" venv="$here/runs/sixcat-venv" max_seconds="$SIXCAT_MAX_SECONDS"
while (($# > 0)); do
  case $1 in
    --url) (($# >= 2)) || die "--url needs a value"; url=$2; shift 2 ;;
    --model) (($# >= 2)) || die "--model needs a value"; model=$2; shift 2 ;;
    --out) (($# >= 2)) || die "--out needs a value"; out=$2; shift 2 ;;
    --venv) (($# >= 2)) || die "--venv needs a value"; venv=$2; shift 2 ;;
    --max-seconds) (($# >= 2)) || die "--max-seconds needs a value"; max_seconds=$2; shift 2 ;;
    -h | --help) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $1 (see --help)" ;;
  esac
done
[[ -n $url ]] || die "--url is required"
[[ -n $model ]] || die "--model is required"
[[ -n $out ]] || die "--out is required"
[[ $max_seconds =~ ^[0-9]+$ && $max_seconds -gt 0 ]] || die "--max-seconds must be a positive integer, got '$max_seconds'"

if [[ "$(cat "$venv/.commit" 2>/dev/null || true)" != "$SIXCAT_COMMIT" ]]; then
  echo "sixcat.sh: installing SixCat $SIXCAT_COMMIT from $SIXCAT_REPO into $venv" >&2
  python3 -m venv "$venv"
  "$venv/bin/pip" install -q "git+$SIXCAT_REPO@$SIXCAT_COMMIT"
  echo "$SIXCAT_COMMIT" > "$venv/.commit"
fi

mkdir -p "$(dirname "$out")"
# No concurrency flag, as in redsnow.sh: its curve stops at C=1 (its server decodes one request at a time, so C>1 only
# queues), and the same command keeps this recipe's score comparable with the other recipes'.
"$venv/bin/python" -m sixcat speed --base-url "$url" --model "$model" --api-key none \
  --policy strict --thinking off --candidates 1 --max-seconds "$max_seconds" --out "$out"
