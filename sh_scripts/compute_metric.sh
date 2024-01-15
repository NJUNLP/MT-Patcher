set -e

infile=${infile:-""}
tokenize=${tokenize:-"13a"}
while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
    fi
    shift
done

echo "####################################################"
echo "####################################################"
echo "Evaluating $infile"
echo "####################################################"
echo "####################################################"

mkdir -p tmp
python3 data_scripts/extract_field.py $infile tmp/hyp.txt tgt_text
python3 data_scripts/extract_field.py $infile tmp/src.txt src_text
python3 data_scripts/extract_field.py $infile tmp/ref.txt reference

echo "####################################################"
echo "SacreBLEU:"
sacrebleu tmp/ref.txt -i tmp/hyp.txt -m bleu -b -w 4 --tokenize $tokenize
echo "####################################################"

echo "####################################################"
echo "COMET score"
comet-score -s tmp/src.txt -t tmp/hyp.txt -r tmp/ref.txt --quiet --only_system
echo "####################################################"

python3 -m bleurt.score_files \
  -candidate_file=tmp/hyp.txt \
  -reference_file=tmp/ref.txt \
  -bleurt_checkpoint=/mnt/bn/st-data-lq/jiahuanli/models/BLEURT-20 \
  -scores_file=tmp/scores
echo "####################################################"
echo "BLEURT Score:"
python3 pipeline/bleurt.py tmp/scores
echo "####################################################"
rm -r tmp