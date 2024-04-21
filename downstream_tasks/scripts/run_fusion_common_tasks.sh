
#!/bin/bash

MODEL_PATHS=; #leave space " " between different models
MODEL_PATHS+="meta-llama/Llama-2-7b-hf "
MODEL_PATHS+="mistralai/Mistral-7B-v0.1 " 
MODEL_PATHS+="microsoft/phi-2 " 
MODEL_PATHS+="Deci/DeciLM-7B "
function join { local IFS=","; echo "$*"; }
model_name=$(join ${MODEL_PATHS[@]})

#TOKENIZER=None #tokenizer selection and alignment

#select fusion models
fusion="opt"  #"sim" "ensemble" "top1"
aggregated_result_file="logs/fusion_${fusion}.txt"

printf "%6s\n" $e >> $aggregated_result_file
for t in "arc-challenge" #"boolq" "hellaswag" "openbookqa"
do
    output_dir="outputs/fusion_${fusion}_common_${t}"
    python main.py --task_name $t --fusion $fusion --model_name $model_name --few_shot 0 --data_cache_dir "datasets" --output_dir $output_dir --annotation_size 5 --seed 1 >> $aggregated_result_file
done
