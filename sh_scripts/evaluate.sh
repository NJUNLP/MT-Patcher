set -e

model=${remote_ckpt_path:-""}
infile=${infile:-""}
savefile=${savefile:-""}
num_gpus=${num_gpus:-""}
srclang=${srclang}
tgtlang=${tgtlang}
while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
    fi
    shift
done

if [[ ! -d `dirname $savefile` ]];then
    mkdir -p `dirname $savefile`
fi

bash sh_scripts/inference.sh \
    --model $model \
    --infile $infile \
    --num_gpus $num_gpus --task translation \
    --output_key tgt_text \
    --srclang $srclang --tgtlang $tgtlang \
    --savefile $savefile

bash sh_scripts/compute_metric.sh --infile $savefile.translation.jsonl

