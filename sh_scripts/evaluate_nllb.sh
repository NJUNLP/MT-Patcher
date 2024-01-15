set -e

model=${remote_ckpt_path:-""}
infile=${infile:-""}
savefile=${savefile:-""}
devices=${devices:-""}
srclang=${srclang}
tgtlang=${tgtlang}
tokenize=${tokenize:-"13a"}
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


python3 pipeline/translation/nllb_generate.py \
    --infile $infile \
    --srclang $srclang --tgtlang $tgtlang \
    --savefile $savefile \
    --model-path $model \
    --devices $devices

bash sh_scripts/compute_metric.sh --infile $savefile --tokenize $tokenize

