
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
for subj in "college_biology" #"college_chemistry" "college_computer_science" "college_mathematics" "college_physics"
do
    output_dir="outputs/fusion_${fusion}_stem_${subj}"
    python main.py --task_name "mmlu" --subj $subj --fusion $fusion --model_name $model_name --few_shot 5 --data_cache_dir "datasets" --output_dir $output_dir --annotation_size 5 >> $aggregated_result_file
done
