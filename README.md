This is the repo for NAACL 2024 paper "MT-Patcher: Selective and Expandable Knowledge Distillation from Large Language Models for Machine Translation" (https://arxiv.org/abs/2403.09522)

# Overview
This repository shares the code of our latest work on efficiently distilling knowledge from LLMs to MT models. In this work,
- We identity two problems in the traditional KD in the LLM era:
    - Non-selective: the student model has mastered reasonable amount of knowledge, while we still repeatedly finetune it on the knowledge its has known in traditional KD
    - Non-extendable: traditional KD cannot transfer knowledge that occurs very few times in the monolingual data very well
- We propose to leveerage LLM's language ability to transfer knowledge more efficiently
    - we convert LLMs to be a translation feedbacker, providing feedbacks on student's translation, and only keep those with errors to train student models
    - we convert LLMs to be a Synthesis Model, synthesizing multiple diverse contexts for errors that student fails on,
    - we convert LLMs to be a Analogy Model, ancipating potential errors from errors that students fails on
- Experiments show that we can achieve comparable translation results by only distilling 10% data, and enlarging the dataset based on the Synthesis Model and Analogy Model further improves the performance.

![20240524-191945](https://github.com/Saltychtao/MT-Patcher/assets/9932507/ce9b2646-a4a7-4495-a8b2-ac9fa7b59567)

# Requirements
Our code is built on huggingface transformers.

- python>=3.8.0
- pytorch>=2.1.0
- vllm (optional)

We recommend to use vllm to accelerate inference.

# MT-Patcher Pipeline

## Step 1: Generate the student's sranslations
For building a MT-Patcher, we need to firstly collect a student's translation of a monolingual corpus. An example scripts is:

```
bash sh_scripts/inference.sh \
  --model wxjiao/ParroT-7b \
  --tokenizer wxjiao/ParroT-7b \
  --infile $monolingual_path \
  --savefile $TRANSLATION_PATH \
  --num_gpus 8 \
  --task translation \
  --output_key tgt_text \
  --use_vllm 1 \
  --srclang en --tgtlang de
```


## Step 2: Collect demonstration data from GPT-4
Next, we need to collect GPT-4's demonstration data for executing the MT-Patcher pipeline. To do this, you should use the prompt provided in the `prompts/` dir, to collect the data from GPT-4.

## Step 3: Train MT-Patcher
After collecting the demonstration data from GPT-4, we can train our MT-Patcher using the following command:

```
bash sh_scripts/deepspeed_run.sh \
  --train_file $COMBINED_GPT4_DATA \
  --model_name_or_path Llama2-13b \
  --batch_size 4 \
  --update_freq 8 \
  --output_dir $PATCHER_MODEL \
  --devices 0,1,2,3,4,5,6,7
```

## Step 4: Run MT-Patcher pipeline on student's translation
We can then run the full MT-Patcher pipeline on student's translation:
```
bash sh_scripts/pipeline.sh \
  --data_file $TRANSLATION_PATH \
  --savefile $PATCH_DATA_PATH \
  --num_gpus 8 \
  --srclang en --tgtlang de \
  --patcher_model $PATCHER_MODEL \
  --generate_feedback 1 --generate_case 1 --generate_analogy 1
```

## Step 5: Finetune the student on the patch data
We can then gather all finetuning data for finetune the student model.

```
bash sh_scripts/deepspeed_run.sh \
  --train_file $COMBINED_PATCH_DATA_PATH \
  --model_name_or_path wxjiao/ParroT-7b \
  --batch_size 4 \
  --update_freq 8 \
  --output_dir $FINETUNED_STUDENT_MODEL \
  --devices 0,1,2,3,4,5,6,7
```
# Citation
If you find this repo useful, please feel free to leave a star and cite our paper:
```
@misc{li2024mtpatcher,
      title={MT-PATCHER: Selective and Extendable Knowledge Distillation from Large Language Models for Machine Translation}, 
      author={Jiahuan Li and Shanbo Cheng and Shujian Huang and Jiajun Chen},
      year={2024},
      eprint={2403.09522},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```



