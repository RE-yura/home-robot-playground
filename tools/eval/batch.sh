#!/bin/zsh
# Usage: batch.sh <label> <query> <seed...>  (learned page served by `pnpm dev` on port $HSR_PORT (default 5183), mug task;
# results under $HSR_RUNS (default work/eval-runs); an existing record is never rerun)
label=$1; query=$2; shift 2
cd "${0:A:h}"
root=${HSR_RUNS:-${0:A:h:h:h}/work/eval-runs}
mkdir -p $root
for seed in "$@"; do
  out=$root/$label-$seed
  [[ -f $out/record.json ]] && continue
  HSR_URL=http://127.0.0.1:${HSR_PORT:-5183} HSR_QUERY="&direct=1$query" node run-task.mjs learned mug $out $seed > $out.log 2>&1 || echo "run $label $seed exited $?" >> $root/errors.log
done
echo "batch $label done"
