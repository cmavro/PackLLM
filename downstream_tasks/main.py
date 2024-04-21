import argparse
import random
import os
import torch
import numpy as np
import json
#import nltk
from tqdm import tqdm
import copy
from sklearn.metrics import f1_score, confusion_matrix

from icl_data import MetaICLData
from icl_model import MetaICLModel

from get_task import get_task
from utils import calculate_sentence_transformer_embedding

from prompt_retrieval import prompt_retrieval

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import  set_seed, InitProcessGroupKwargs
from datetime import timedelta
import pickle

class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "train_clusterer"
        return super().find_class(module, name)


def load_model(path_to_model):
    with open(path_to_model, 'rb') as f:
        unpickler = MyCustomUnpickler(f)
        out = unpickler.load()
    return out


parser = argparse.ArgumentParser()
parser.add_argument('--task_name', required=True,type=str)
parser.add_argument('--model_cache_dir', required=False,type=str)
parser.add_argument('--data_cache_dir', required=True,type=str)
parser.add_argument('--output_dir', required=True,type=str)
parser.add_argument('--prompt_retrieval_method', default='similar',type=str)
parser.add_argument('--model_name', default='EleutherAI/gpt-j-6B',type=str)
parser.add_argument('--embedding_model', default='sentence-transformers/all-mpnet-base-v2',type=str)
parser.add_argument('--annotation_size', default=100,type=int)
parser.add_argument('--seed', default=0,type=int)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--annotation_path',default='first_phase_selected_indices',type=str)
parser.add_argument('--few_shot',default=5,type=int) #0 means we concat as much as possible
parser.add_argument('--init_size',default=10,type=int) 
parser.add_argument('--sample_k',action='store_true')
parser.add_argument('--mixed_precision',default="fp16",type=str) 
parser.add_argument("--subj",type=str,default=None) #mmlu

parser.add_argument('--tokenizer_name',default=None,type=str)  #None means no reference tokenizer (will be top1)
parser.add_argument("--fusion", type=str, default="opt", choices=["opt", "sim", "top1", "ensemble", "dexperts", "dexperts-opt"]) #fusion method



args = parser.parse_args()
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__=='__main__':
    set_seed(args.seed)
    #for retrieval
    default_embedding_model = 'sentence-transformers/all-mpnet-base-v2'
    args.output_dir += f"/single_phase_few_shot-{args.few_shot}/{args.task_name}_lm-{args.model_name}_results"
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=6000000))
    accelerator = (
        Accelerator(mixed_precision=args.mixed_precision, kwargs_handlers=[kwargs],)
    )
    accelerator.wait_for_everyone()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir,exist_ok=True)

    with open(os.path.join(args.output_dir,'result_summary.txt'), 'w') as f:
        f.write(f"{args.output_dir}\n")

    print("\n")
    print("=====================")
    print("DATASET: ", args.task_name)
    print("=====================")
    print("\n")
    if args.embedding_model != default_embedding_model:
        new_embedding_model = args.embedding_model
        args.embedding_model = default_embedding_model
        train_examples,eval_examples,train_text_to_encode,eval_text_to_encode,format_example,label_map = get_task(args=args)
        args.embedding_model = new_embedding_model
    else:
        train_examples,eval_examples,train_text_to_encode,eval_text_to_encode,format_example,label_map = get_task(args=args)

    #train / test pools of ICL examples
    if accelerator.is_main_process:
        total_train_embeds = calculate_sentence_transformer_embedding(text_to_encode=train_text_to_encode,
                                                                    args=args, accelerator=accelerator)
        total_eval_embeds = calculate_sentence_transformer_embedding(text_to_encode=eval_text_to_encode,
                                                                    args=args, accelerator=accelerator)
        output_dir_examples = os.path.join(args.output_dir,'examples')
        if not os.path.isdir(output_dir_examples):
            os.makedirs(output_dir_examples, exist_ok=True)

        path = os.path.join(output_dir_examples, 'all_train_examples.json')
        with open(path, 'w') as fout:
            json.dump(train_examples, fout, indent=4)

        path = os.path.join(output_dir_examples, 'all_eval_examples.json')
        with open(path, 'w') as fout:
            json.dump(eval_examples, fout, indent=4)

    accelerator.wait_for_everyone()

    #create tokenizers for input data (or reference "tokenizer_name")
    if args.task_name in ['hellaswag', 'boolq', 'openbookqa', 'arc-challenge', 'pubmedqa', 'usmle']: #long examples 0-shot
        max_length_per_example = 2048
    else:
        max_length_per_example = 340
    tokenizer_name = args.tokenizer_name
    cur_max_size = 1
    data_module = []
    model_path_list = args.model_name.split(",")
    for model_name in model_path_list:
        data_module.append(MetaICLData(method="direct", max_length=2048, max_length_per_example=max_length_per_example, tokenizer_name=tokenizer_name, model_name=model_name))
        if len(data_module[-1].tokenizer) > cur_max_size:
            prompt_tokenizer = data_module[-1].tokenizer
            cur_max_size = len(data_module[-1].tokenizer)
    
    
    #create packllm model
    print("Model using", args.model_name)
    inference_model = MetaICLModel(args=args, accelerator=accelerator)
    inference_model.load(fusion=args.fusion, tokenizer=args.tokenizer_name)
    inference_model.model.accelerator = accelerator
    inference_model.eval()


    #prompt construction with few shot examples
    if accelerator.is_main_process:
        path = os.path.join(output_dir_examples, "_"+"B"+str(args.annotation_size)+'selected_train_examples.json')
        #random few shot examples
        random.seed(args.seed) 
        first_phase_selected_indices =  random.sample(range(len(train_examples)), args.annotation_size)
        with open(os.path.join(args.output_dir, f"selected_indices_final.json"),'w') as f:
            json.dump(first_phase_selected_indices,f,indent=4)
        processed_train_examples = [train_examples[idx] for idx in first_phase_selected_indices]
        with open(path, 'w') as fout:
            json.dump(processed_train_examples, fout, indent=4)
        processed_eval_examples = eval_examples

        return_string = False

        single_input_len = 350
        maximum_input_len = 2048
        prompt_retrieval(train_embs=total_train_embeds[first_phase_selected_indices],test_embs=total_eval_embeds,train_examples=processed_train_examples,
                            eval_examples=processed_eval_examples,return_string=False,format_example=format_example,
                            maximum_input_len=maximum_input_len,single_context_example_len=single_input_len,label_map=label_map,args=args, prompt_tokenizer=prompt_tokenizer)

            

    accelerator.wait_for_everyone()

    prompt_cache_dir = os.path.join(args.output_dir, 'prompts')
    candidate_prompt_files = os.listdir(prompt_cache_dir)
    prompt_files = [f for f in candidate_prompt_files if f.endswith('.json')]
    if accelerator.is_main_process:
        assert len(prompt_files) == len(processed_eval_examples), f"len(prompt_files)={len(prompt_files)}," \
                                                                f"len(processed_eval_examples)={len(processed_eval_examples)}"
    accelerator.wait_for_everyone()
        

    output_dir = os.path.join(args.output_dir,'results_final_test')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    golds = []
    preds = []
    scores = []
        

    #inference code and evaluation    
    bar = tqdm(range(len(prompt_files)), desc=f"  LLM inference")
    for file in prompt_files:
        bar.update(1)
                
        with open(os.path.join(prompt_cache_dir, file)) as f:
            one_test_example = json.load(f)
        cur_train_data = one_test_example[1]
        formatted = format_example(one_test_example[2], label_map=label_map, args=args)
        prompt = formatted[2]
        cur_input = {'input': formatted[0],
                    'options': one_test_example[2]['endings']}
        tokenizer_id = inference_model.model.select_tokenizer([one_test_example[2]['text']])
        
        data_module[tokenizer_id].k = len(cur_train_data)
        data_module[tokenizer_id].tensorize(cur_train_data, [cur_input], prompt = [prompt])
        inference_model.model.align_tokenizers(tokenizer_id) 
        inference_model.model.set_expert(tokenizer_id)
        prediction = inference_model.do_predict(data_module[tokenizer_id], require_loss=True)[0]
                                
        if  prediction[0] not in one_test_example[2]['endings']:
            #print(f"Wrong prediction {prediction[0]}, choosing at random \n")
            prediction[0] = one_test_example[2]['endings'][0]
        with open(f"{output_dir}/{file}", 'w') as f:
            try: 
                json.dump([prediction[0], one_test_example[2]['endings'][one_test_example[2]['label']]], f)
                golds.append(one_test_example[2]['endings'][one_test_example[2]['label']])
            except:
                json.dump([prediction[0], one_test_example[2]['label']], f)
                golds.append(one_test_example[2]['label'])
        preds.append(prediction[0])
        scores.append(prediction[1])
        
    #final evaluation
    if accelerator.is_main_process:
        results = []
        assert len(golds) == len(preds), f"len(golds)={len(golds)}, len(preds)={len(preds)}"
        total = len(golds)
        correct = 0
        for p, g in zip(golds, preds):
            if p == g:
                correct += 1
                results.append(1)
            else:
                results.append(0)
        
        with open(os.path.join(args.output_dir,'result_summary.txt'), 'a') as f:
            f.write(f"Models: {args.model_name}\n")
            f.write(f"{len(golds)} examples, accuracy is: {correct / total}\n")

        print(f'The accuracy score is {correct / total}\n')
