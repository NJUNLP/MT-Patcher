set -e

data_file=${data_file:-""}
savefile=${savefile:-""}
max_tokens=${max_tokens:-"512"}
num_gpus=${num_gpus:-""}

srclang=${srclang:-""}
tgtlang=${tgtlang:-""}

patcher_model=${patcher_model:-""}
# misc_model=${misc_model:-""}

generate_feedback=${generate_feedback:-"1"}
generate_case=${generate_case:-"1"}
generate_analogy=${generate_analogy:-""}


while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
    fi
    shift
done

echo `dirname $savefile`
if [[ ! -f `dirname $savefile` ]];then
    mkdir -p `dirname $savefile`
fi

echo $patcher_model
assessment_out=$savefile.feedback_parsed.json
if [[ $generate_feedback == "1" ]];then
echo "Generating Feedback"
echo $assessment_out
bash sh_scripts/inference.sh \
    --model ${patcher_model} \
    --infile ${data_file} \
    --savefile $savefile \
    --num_gpus $num_gpus \
    --task feedback \
    --srclang $srclang --tgtlang $tgtlang
fi

sa_out=$savefile.sentence_analysis.jsonl
wa_out=$savefile.word_analogy.jsonl
cg_out=$savefile.case_generation.jsonl
if [[ $generate_case == "1" ]];then
    echo "Generating SA"
    bash sh_scripts/inference.sh \
        --model ${patcher_model} \
        --infile ${assessment_out} \
        --savefile $savefile \
        --num_gpus $num_gpus \
        --task sentence_analysis \
        --srclang $srclang --tgtlang $tgtlang

    if [[ $generate_analogy == "1" ]];then
        echo "Generating WA"
        bash sh_scripts/inference.sh \
            --model ${patcher_model} \
            --infile ${sa_out} \
            --savefile $savefile \
            --num_gpus $num_gpus \
            --task word_analoger \
            --srclang $srclang --tgtlang $tgtlang

        echo "Generating Case"
        bash sh_scripts/inference.sh \
            --model ${patcher_model} \
            --infile ${wa_out} \
            --savefile $savefile \
            --num_gpus $num_gpus \
            --task case_generation_from_word_analogy \
            --srclang $srclang --tgtlang $tgtlang
    else
        echo "Generating Case"
        bash sh_scripts/inference.sh \
            --model ${patcher_model} \
            --infile ${sa_out} \
            --savefile $savefile \
            --num_gpus $num_gpus \
            --task case_generation \
            --srclang $srclang --tgtlang $tgtlang
    fi
fi