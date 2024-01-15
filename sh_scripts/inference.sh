set -e

model=${remote_ckpt_path:-""}
tokenizer=${tokenizer:-""}
infile=${infile:-""}
savefile=${savefile:-""}
num_gpus=${num_gpus:-""}
task=${task:-""}
output_key=${output_key:-""}
use_vllm=${use_vllm:-"1"}

srclang=${srclang:-""}
tgtlang=${tgtlang:-""}
while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
    fi
    shift
done

echo $infile
echo $savefile
echo $num_gpus
echo $task

echo `dirname $savefile`
if [[ ! -f `dirname $savefile` ]];then
    mkdir -p `dirname $savefile`
fi

mkdir -p tmp

if [[ $use_vllm == "0" ]];then
    vllm_args=""
else
    vllm_args="--use-vllm"
fi

if [[ $output_key == "" ]];then
    output_key_args=""
else
    output_key_args="--output-key ${output_key}"
fi

if [[ -f $savefile.$task.jsonl ]]; then
    rm $savefile.$task.jsonl
fi

if [[ $tokenizer == "" ]];then
    tokenizer=$model
fi

if [[ $num_gpus == "1" ]]; then
    python3 pipeline/inference.py \
            --model-path ${model} \
            --config-dir ${model} \
            --savefile $savefile.$task.jsonl \
            --infile $infile \
            --devices 0  $output_key_args\
            --task $task $vllm_args \
            --tokenizer $tokenizer \
            --srclang $srclang --tgtlang $tgtlang 
else
    ((num_gpus_minus_1 = $num_gpus - 1 ))
    python3 pipeline/split.py $infile $num_gpus
    for i in `seq 0 ${num_gpus_minus_1}`; do
        echo "starting process $i"
        CUDA_VISIBLE_DEVICES=$i python3 pipeline/inference.py \
            --model-path ${model} \
            --config-dir ${model} \
            --savefile tmp/chunk${i}.output.jsonl \
            --infile tmp/chunk${i}.jsonl \
            --devices 0  $output_key_args\
            --tokenizer $tokenizer \
            --srclang $srclang --tgtlang $tgtlang \
            --task $task $vllm_args &
    done
    wait
    cat tmp/chunk*.output.jsonl >> $savefile.$task.jsonl
fi


if [[ $task == "feedback" ]];then
    python3 pipeline/assessement/postprocess.py --infile $savefile.feedback.jsonl --outfile $savefile.feedback_parsed.json
fi

rm -rf tmp
