#!/usr/bin/env bash
set -euo pipefail
root=/opt/qwen38-910
backend=${1:-CANN4}
[[ "$backend" =~ ^CANN[0-9]+$ ]] || { echo 'Expected a backend such as CANN4' >&2; exit 2; }
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 "$root/tests/oracles/gdn_reference.py" --out "$temporary/gdn" --value-heads 48 > "$temporary/gdn.json"
"$root/bin/test_gdn_backend" "$temporary/gdn" "$backend" "$root/lib" 48
python3 "$root/tests/oracles/ssm_conv_reference.py" --out "$temporary/conv" --tokens 1 > "$temporary/conv.json"
"$root/bin/test_ssm_bench" "$temporary/conv" 1 "$backend" "$root/lib" 0.000002 0.000002
for shape in '5120 1 5120' '128 48 128' '256 24 512' '256 4 256'; do
  # Fixed, repository-owned numeric tuples; no user input is evaluated here.
  read -r width rows stride <<< "$shape"
  "$root/bin/test_rms_backend" "$width" "$rows" "$stride" "$backend" "$root/lib"
done
"$root/bin/test_rms_backend" 128 16 128 "$backend" "$root/lib" 7.8125e-9
"$root/bin/test_rope_backend" "$backend" "$root/lib"
python3 "$root/tests/oracles/generate_fixtures.py" --out "$temporary/setrows" > "$temporary/setrows.json"
"$root/bin/test_setrows_backend" "$backend" "$root/lib" "$temporary/setrows"
echo 'All selected hardware operator tests passed'
