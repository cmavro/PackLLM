import os
import json
import random
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
import copy


def format_dataset(sample):
    question = sample['question']['text']
    context = sample['document']['tokens']['token']
    is_html = sample['document']['tokens']['is_html']
    long_answers = sample['annotations']['long_answer']
    short_answers = sample['annotations']['short_answers']

    context_string =  " ".join([context[i] for i in range(len(context)) if not is_html[i]])

    # 0 - No ; 1 - Yes
    for answer in sample['annotations']['yes_no_answer']:
        if answer == 0 or answer == 1:
            return {"question": question, "short": ["no" if answer == 0 else "yes"], "long": [], "category": "no" if answer == 0 else "yes"}

    short_targets = []
    for s in short_answers:
        short_targets.extend(s['text'])
    short_targets = list(set(short_targets))

    long_targets = []
    for s in long_answers:
        if s['start_token'] == -1:
            continue
        answer = context[s['start_token']: s['end_token']]
        html = is_html[s['start_token']: s['end_token']]
        new_answer = " ".join([answer[i] for i in range(len(answer)) if not html[i]])
        if new_answer not in long_targets:
            long_targets.append(new_answer)

    category = "other" if len(short_targets) > 0 else "null"

    return {"question": question, "short": short_targets, "long": long_targets, "category": category}



def process_mmlu_examples(examples):
    processed_examples = []
    idx = 0
    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    
    for raw_data in tqdm(examples,desc='process mmlu examples'):
        #print(raw_data)
        if "input" in raw_data:
            processed_examples.append({
                'id': idx,
                'text': raw_data['input'],
                'endings': [raw_data['A'],  raw_data['B'], raw_data['C'], raw_data['D']],
                'label': label_map[raw_data['target']]
            })
        elif "question" in raw_data:
            assert len(raw_data['choices']) == 4
            #label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'D'}
            processed_examples.append({
                'id': idx,
                'text': raw_data['question'],
                'endings': ["A. " + raw_data['choices'][0] , "B. " + raw_data['choices'][1], "C. " + raw_data['choices'][2], "D. " + raw_data['choices'][3]],
                'label': raw_data['answer']
            })
        else:
            processed_examples.append({
                'id': idx,
                'text': raw_data['text'],
                'endings': raw_data['endings'],
                'label': raw_data['label']
            })
        idx += 1
    return processed_examples

def process_openbookqa_examples(examples):
    processed_examples = []
    idx = 0
    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    
    for raw_data in tqdm(examples,desc='process openbookqa examples'):
        #print(raw_data)
        if "question_stem" in raw_data:
            processed_examples.append({
                'id': idx,
                'text': raw_data['question_stem'],
                'endings': ["A. " + raw_data['choices']['text'][0],  "B. " + raw_data['choices']['text'][1], "C. " + raw_data['choices']['text'][2], "D. " + raw_data['choices']['text'][3]],
                'label': label_map[raw_data['answerKey']]
            })
        else:
            processed_examples.append({
                'id': idx,
                'text': raw_data['text'],
                'endings': raw_data['endings'],
                'label': raw_data['label']
            })
        idx += 1
    return processed_examples


def process_arc_examples(examples):
    processed_examples = []
    idx = 0
    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, '1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
    
    for raw_data in tqdm(examples,desc='process arc examples'):
        #print(raw_data)
        if "question" in raw_data:
            
            if len(raw_data['choices']['text']) == 3:
                processed_examples.append({
                    'id': idx,
                    'text': raw_data['question'],
                    'endings': ["A. "+ raw_data['choices']['text'][0], "B. "+ raw_data['choices']['text'][1], "C. "+ raw_data['choices']['text'][2]],
                    'label': label_map[raw_data['answerKey']]
                })
            elif len(raw_data['choices']['text']) == 4:
                processed_examples.append({
                    'id': idx,
                    'text': raw_data['question'],
                    'endings': ["A. "+ raw_data['choices']['text'][0], "B. "+ raw_data['choices']['text'][1], "C. "+ raw_data['choices']['text'][2],  "D. "+ raw_data['choices']['text'][3]],
                    'label': label_map[raw_data['answerKey']]
                })
            elif  len(raw_data['choices']['text']) == 5:
                processed_examples.append({
                    'id': idx,
                    'text': raw_data['question'],
                    'endings': ["A. "+ raw_data['choices']['text'][0], "B. "+ raw_data['choices']['text'][1], "C. "+ raw_data['choices']['text'][2],  "D. "+ raw_data['choices']['text'][3], "E. "+ raw_data['choices']['text'][4]],
                    'label': label_map[raw_data['answerKey']]
                })
        else:
            processed_examples.append({
                'id': idx,
                'text': raw_data['text'],
                'endings': raw_data['endings'],
                'label': raw_data['label']
            })
        idx += 1

    return processed_examples


def process_hellaswag_examples(examples):
    processed_examples = []
    idx = 0
    for raw_data in tqdm(examples,desc='process hellaswag examples'):
        processed_examples.append({
            'id': idx,
            'ctx_a': raw_data['ctx_a'],
            'ctx_b': raw_data['ctx_b'],
            'ctx':raw_data['ctx'],
            'text': raw_data['ctx'],
            'endings':raw_data['endings'],
            'label':int(raw_data['label']),
            'activity_label':raw_data['activity_label']
        })
        idx += 1
    return processed_examples

def process_boolq_examples(examples):
    processed_examples = []
    idx = 0
    label_map = {'true': 'Yes', 'false':'No', 'True': 'Yes', 'False':'No'}
    for raw_data in tqdm(examples,desc='process boolq examples'):
        #print(raw_data['answer'])
        processed_examples.append({
            'id': idx,
            'question': raw_data['question'],
            'passage': raw_data['passage'],
            'text': raw_data['passage'],
            'endings': ['Yes', 'No'],
            'label': label_map[str(raw_data['answer'])],
        })
        idx += 1
    return processed_examples

def process_agnews_examples(examples):
    processed_examples = []
    idx = 0
    labels = ["world", "sports", "business", "science"]
    for raw_data in tqdm(examples,desc='process ag_news examples'):
        processed_examples.append({
            'id': idx,
            'label': labels[int(raw_data['label'])],
            'text': raw_data['text'],
            'endings': labels.copy(),
        })
        idx += 1
    return processed_examples

def process_sst2_examples(examples):
    processed_examples = []
    idx = 0
    labels = ["negative", "positive"]
    for raw_data in tqdm(examples,desc='process sst2 examples'):
        processed_examples.append({
            'id': idx,
            'label': labels[int(raw_data['label'])],
            'text': raw_data['sentence'],
            'endings': labels.copy(),
        })
        idx += 1
    return processed_examples

def process_ethos_examples(examples):
    processed_examples = []
    idx = 0
    labels = ["acceptable", "hateful"]
    for raw_data in tqdm(examples,desc='process ethos examples'):
        processed_examples.append({
            'id': idx,
            'label': labels[int(raw_data['label'])],
            'text': raw_data['text'],
            'endings': labels.copy(),
        })
        idx += 1
    return processed_examples

def process_tweet_hate_examples(examples):
    processed_examples = []
    idx = 0
    labels = ["acceptable", "hateful"]
    for raw_data in tqdm(examples,desc='process tweet_hate examples'):
        processed_examples.append({
            'id': idx,
            'label': labels[int(raw_data['label'])],
            'text': raw_data['text'],
            'endings': labels.copy(),
        })
        idx += 1
    return processed_examples

def process_amazon_examples(examples):
    processed_examples = []
    idx = 0
    labels = ["negative","positive"]
    for raw_data in tqdm(examples,desc='process amazon polarity examples'):
        processed_examples.append({
            'id': idx,
            'label':  labels[int(raw_data['label'])],
            'text': raw_data['content'],
            'endings': labels.copy(),
        })
        idx += 1
    return processed_examples

def process_pubmedqa_examples(examples):
    processed_examples = []
    idx = 0
    
    for raw_data in tqdm(examples,desc='process pubmedqa examples'):
        #print(raw_data['answer'])
        processed_examples.append({
            'id': idx,
            'text': raw_data['input'],
            'endings': raw_data['options'],
            'label': raw_data['options'][raw_data['gold_index']],
        })
        idx += 1
    return processed_examples



def process_usmle_examples(examples):
    processed_examples = []
    idx = 0
    
    for raw_data in tqdm(examples,desc='process usmle examples'):
        #print(raw_data['answer'])
        processed_examples.append({
            'id': idx,
            'text': raw_data['input'],
            'endings': raw_data['options'],
            'label': raw_data['options'][raw_data['gold_index']],
        })
        idx += 1
    return processed_examples

def process_complaints_examples(examples):
    processed_examples = []
    idx = 0

    labels = ["complaint", "no complaint"]
    for raw_data in tqdm(examples,desc='process twitter complaints examples'):

        if "Tweet text" in raw_data:
            processed_examples.append({
                'id': idx,
                'label': labels[int(raw_data['Label']) - 1],
                'text': raw_data['Tweet text'],
                'endings': labels.copy(),
            })
        else:
            processed_examples.append({
                'id': idx,
                'label': raw_data['label'],
                'text': raw_data['text'],
                'endings': raw_data['endings'],
            })

        idx += 1
    return processed_examples


def get_task(args):
    task_name = args.task_name
    data_cache_dir = args.data_cache_dir
    #print("New prompt")
    if 'mmlu' in task_name:
        if False: #os.path.isfile(os.path.join(args.output_dir,f'train_examples_seed_{args.seed}.json')) and \
            #os.path.isfile(os.path.join(args.output_dir,f'eval_examples_seed_{args.seed}.json')):
            print('use cached examples')
            with open(os.path.join(args.output_dir,f'train_examples_seed_{args.seed}.json')) as f:
                total_train_examples = json.load(f)
            with open(os.path.join(args.output_dir,f'eval_examples_seed_{args.seed}.json')) as f:
                total_eval_examples = json.load(f)
        else:
            assert args.subj is not None

            print(f"MMLU Task\nSubject {args.subj}\n\n")
            datasets = load_dataset('cais/mmlu', args.subj, cache_dir=data_cache_dir)
            total_train_examples = [e for e in datasets['dev']]
            total_train_examples = process_mmlu_examples(total_train_examples)
            total_eval_examples = [e for e in datasets['test']]
            total_eval_examples = process_mmlu_examples(total_eval_examples)
            if 'val' in task_name:
                total_train_examples = [e for e in datasets['validation']]
                
                total_train_examples = process_mmlu_examples(total_train_examples)

           
            #total_eval_examples = random.sample(total_eval_examples, 256)
            
            with open(os.path.join(args.output_dir,f'train_examples_seed_{args.seed}.json'),'w') as f:
                json.dump(total_train_examples,f,indent=4)
            with open(os.path.join(args.output_dir,f'eval_examples_seed_{args.seed}.json'),'w') as f:
                json.dump(total_eval_examples,f,indent=4)
        if args.debug:
            args.annotation_size = 10
            args.batch_size = 1
            total_train_examples = total_train_examples[:50]
            total_eval_examples = total_eval_examples[:5]
        def format_example(example,label_map,**kwargs):
            prompt = "\nAnswer:"
            label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'D'}
            #print(f"Question: {example['text'].strip()}\nChoices:\n{example['endings'][0]}\n{example['endings'][1]}\n{example['endings'][2]}\n{example['endings'][3]}")
            #print( f"{example['endings'][example['label']]}", prompt)
            return f"Question: {example['text'].strip()}\nChoices:\n{example['endings'][0]}\n{example['endings'][1]}\n{example['endings'][2]}\n{example['endings'][3]}", f"{example['endings'][example['label']]}", prompt
            # return f"{example['premise']}. Based on that information, is the claim {example['hypothesis']} \"True\", " \
            #    f"\"False\", or \"Inconclusive\"?\nanswer:", f"{label_map[example['label']]}"

       
        all_train_text_to_encode = ["{}".format(raw_item["text"]) for raw_item in total_train_examples]
        
        all_eval_text_to_encode = ["{}".format(raw_item["text"]) for raw_item in total_eval_examples]
        #label_map = {0:"True",1:"Inconclusive",2:"False"}
        
        #label_map = {'A':"A",'B':"B", 'C':"C", 'D':"D"}
        label_map = None

    elif task_name=='pubmedqa':
        if False: # if os.path.isfile(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) and \
        #         os.path.isfile(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')):
            print('use cached examples')
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) as f:
                total_train_examples = json.load(f)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')) as f:
                total_eval_examples = json.load(f)
        else:
            datasets = load_dataset('AdaptLLM/medicine-tasks', 'PubMedQA', cache_dir=data_cache_dir)
            total_train_examples = [e for e in datasets['test']]
            total_train_examples = random.sample(total_train_examples, 310)
            total_train_examples = process_pubmedqa_examples(total_train_examples)
            total_eval_examples = [e for e in datasets['test']]
            total_eval_examples = random.sample(total_eval_examples, 256)
            total_eval_examples = process_pubmedqa_examples(total_eval_examples)
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_train_examples, f, indent=4)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_eval_examples, f, indent=4)
        if args.debug:
            args.annotation_size = 10
            args.batch_size = 1
            total_train_examples = total_train_examples[:50]
            total_eval_examples = total_eval_examples[:5]
        def format_example(example,label_map,**kwargs):
            #label_map = {'true': 'Yes', 'false':'No'}
            prompt = "\nAnswer: "
            return f"{example['text']}", \
                   f"{example['label']}", prompt
        
        all_train_text_to_encode = [f"{raw_item['text']}" for raw_item in total_train_examples]
        all_eval_text_to_encode = [f"{raw_item['text']}" for raw_item in total_eval_examples]
        label_map = None

    elif task_name=='usmle':
        if False: # if os.path.isfile(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) and \
        #         os.path.isfile(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')):
            print('use cached examples')
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) as f:
                total_train_examples = json.load(f)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')) as f:
                total_eval_examples = json.load(f)
        else:
            datasets = load_dataset('AdaptLLM/medicine-tasks', 'USMLE', cache_dir=data_cache_dir)
            total_train_examples = [e for e in datasets['test']]
            total_train_examples = random.sample(total_train_examples, 310)
            total_train_examples = process_usmle_examples(total_train_examples)
            total_eval_examples = [e for e in datasets['test']]
            total_eval_examples = random.sample(total_eval_examples, 256)
            total_eval_examples = process_usmle_examples(total_eval_examples)
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_train_examples, f, indent=4)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_eval_examples, f, indent=4)
        if args.debug:
            args.annotation_size = 10
            args.batch_size = 1
            total_train_examples = total_train_examples[:50]
            total_eval_examples = total_eval_examples[:5]
        def format_example(example,label_map,**kwargs):
            #label_map = {'true': 'Yes', 'false':'No'}
            prompt = " "
            return f"{example['text']}", \
                   f"{example['label']}", prompt
        
        all_train_text_to_encode = [f"{raw_item['text']}" for raw_item in total_train_examples]
        all_eval_text_to_encode = [f"{raw_item['text']}" for raw_item in total_eval_examples]
        label_map = None


    elif task_name=='openbookqa':
        if False: # if os.path.isfile(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) and \
        #         os.path.isfile(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')):
            print('use cached examples')
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) as f:
                total_train_examples = json.load(f)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')) as f:
                total_eval_examples = json.load(f)
        else:
            datasets = load_dataset('allenai/openbookqa', 'main', cache_dir=data_cache_dir)
            total_train_examples = [e for e in datasets['train']]
            total_train_examples = random.sample(total_train_examples, 310)
            total_train_examples = process_openbookqa_examples(total_train_examples)
            total_eval_examples = [e for e in datasets['validation']]
            total_eval_examples = random.sample(total_eval_examples, 256)
            total_eval_examples = process_openbookqa_examples(total_eval_examples)
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_train_examples, f, indent=4)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_eval_examples, f, indent=4)
        if args.debug:
            args.annotation_size = 10
            args.batch_size = 1
            total_train_examples = total_train_examples[:50]
            total_eval_examples = total_eval_examples[:5]
        def format_example(example,label_map,**kwargs):
            #label_map = {'true': 'Yes', 'false':'No'}
            prompt = "\nAnswer:"
            label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
            return f"Question: {example['text'].strip()}\nChoices:\n{example['endings'][0]}\n{example['endings'][1]}\n{example['endings'][2]}\n{example['endings'][3]}", f"{example['endings'][example['label']]}", prompt
        
        all_train_text_to_encode = ["{}".format(raw_item["text"]) for raw_item in total_train_examples]
        all_eval_text_to_encode = ["{}".format(raw_item["text"]) for raw_item in total_eval_examples]
        label_map = None

    elif 'arc' in task_name:
        if False: # if os.path.isfile(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) and \
        #         os.path.isfile(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')):
            print('use cached examples')
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) as f:
                total_train_examples = json.load(f)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')) as f:
                total_eval_examples = json.load(f)
        else:
            if 'easy' in task_name:
                datasets = load_dataset('allenai/ai2_arc', 'ARC-Easy', cache_dir=data_cache_dir)
            elif 'challenge' in task_name:
                datasets = load_dataset('allenai/ai2_arc', 'ARC-Challenge', cache_dir=data_cache_dir)
            total_train_examples = [e for e in datasets['train']]
            total_train_examples = random.sample(total_train_examples, 310)
            total_train_examples = process_arc_examples(total_train_examples)
            total_eval_examples = [e for e in datasets['validation']]
            total_eval_examples = random.sample(total_eval_examples, 256)
            total_eval_examples = process_arc_examples(total_eval_examples)
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_train_examples, f, indent=4)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_eval_examples, f, indent=4)
        if args.debug:
            args.annotation_size = 10
            args.batch_size = 1
            total_train_examples = total_train_examples[:50]
            total_eval_examples = total_eval_examples[:5]
        def format_example(example,label_map,**kwargs):
            #label_map = {'true': 'Yes', 'false':'No'}
            prompt = "\nAnswer:"
            label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
            if len(example['endings']) == 3:

                return f"Question: {example['text'].strip()}\nChoices:\n{example['endings'][0]}\n{example['endings'][1]}\n{example['endings'][2]}", f"{example['endings'][example['label']]}", prompt
            elif len(example['endings']) == 4:
                return f"Question: {example['text'].strip()}\nChoices:\n{example['endings'][0]}\n{example['endings'][1]}\n{example['endings'][2]}\n{example['endings'][3]}", f"{example['endings'][example['label']]}", prompt
            elif len(example['endings']) == 5:
                return f"Question: {example['text'].strip()}\nChoices:\n{example['endings'][0]}\n{example['endings'][1]}\n{example['endings'][2]}\n{example['endings'][3]}\n{example['endings'][4]}", f"{example['endings'][example['label']]}", prompt

        all_train_text_to_encode = ["{}".format(raw_item["text"]) for raw_item in total_train_examples]
        all_eval_text_to_encode = ["{}".format(raw_item["text"]) for raw_item in total_eval_examples]
        label_map = None

    elif task_name=='hellaswag':
        if False: #os.path.isfile(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) and \
            #    os.path.isfile(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')):
            print('use cached examples')
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) as f:
                total_train_examples = json.load(f)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')) as f:
                total_eval_examples = json.load(f)
        else:
            datasets = load_dataset('Rowan/hellaswag',cache_dir=data_cache_dir)
            total_train_examples = [e for e in datasets['train']]
            total_train_examples = random.sample(total_train_examples, 310)
            total_train_examples = process_hellaswag_examples(total_train_examples)
            total_eval_examples = [e for e in datasets['validation']]
            total_eval_examples = random.sample(total_eval_examples, 256)
            total_eval_examples = process_hellaswag_examples(total_eval_examples)
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_train_examples, f, indent=4)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_eval_examples, f, indent=4)
        if args.debug:
            args.annotation_size = 10
            args.batch_size = 1
            total_train_examples = total_train_examples[:50]
            total_eval_examples = total_eval_examples[:5]
        def format_example(example,label_map,**kwargs):
            return f"The topic is {example['activity_label']}. {example['ctx_a']} " \
                   f"{example['ctx_b']}", f"{example['endings'][example['label']]}", " "

        all_train_text_to_encode = [f"The topic is {raw_item['activity_label']}. {raw_item['ctx_a']} {raw_item['ctx_b']} | " \
                                  f"{raw_item['endings'][0]} | " \
                                  f"{raw_item['endings'][1]} | " \
                                  f"{raw_item['endings'][2]} | " \
                                  f"{raw_item['endings'][3]}" for raw_item in total_train_examples]
        all_eval_text_to_encode = [f"The topic is {raw_item['activity_label']}. {raw_item['ctx_a']} {raw_item['ctx_b']} | " \
                                  f"{raw_item['endings'][0]} | " \
                                  f"{raw_item['endings'][1]} | " \
                                  f"{raw_item['endings'][2]} | " \
                                  f"{raw_item['endings'][3]}" for raw_item in total_eval_examples]
        label_map = None

    elif task_name=='boolq':
        if False: # if os.path.isfile(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) and \
        #         os.path.isfile(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')):
            print('use cached examples')
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) as f:
                total_train_examples = json.load(f)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')) as f:
                total_eval_examples = json.load(f)
        else:
            boolq_datasets = load_dataset('google/boolq',cache_dir=data_cache_dir)
            total_train_examples = [e for e in boolq_datasets['train']]
            total_train_examples = random.sample(total_train_examples, 310)
            total_train_examples = process_boolq_examples(total_train_examples)
            total_eval_examples = [e for e in boolq_datasets['validation']]
            total_eval_examples = random.sample(total_eval_examples, 256)
            total_eval_examples = process_boolq_examples(total_eval_examples)
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_train_examples, f, indent=4)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_eval_examples, f, indent=4)
        if args.debug:
            args.annotation_size = 10
            args.batch_size = 1
            total_train_examples = total_train_examples[:50]
            total_eval_examples = total_eval_examples[:5]
        def format_example(example,label_map,**kwargs):
            #label_map = {'true': 'Yes', 'false':'No'}
            prompt = "\nAnswer: "
            return f"{example['passage']}\nQuestion: {example['question']}?", \
                   f"{example['label']}", prompt
        
        all_train_text_to_encode = [f"{raw_item['passage']}" for raw_item in total_train_examples]
        all_eval_text_to_encode = [f"{raw_item['passage']}" for raw_item in total_eval_examples]
        label_map = None

    elif task_name=='ag_news':
        if False: #os.path.isfile(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) and \
            #    os.path.isfile(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')):
            print('use cached examples')
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) as f:
                total_train_examples = json.load(f)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')) as f:
                total_eval_examples = json.load(f)
        else:
            datasets = load_dataset('ag_news', cache_dir=data_cache_dir)
            total_train_examples = [e for e in datasets['train']]
            total_train_examples = random.sample(total_train_examples, 310)

            total_train_examples = process_agnews_examples(total_train_examples)
            total_eval_examples = [e for e in datasets['test']]
            total_eval_examples = random.sample(total_eval_examples, 256)
            total_eval_examples = process_agnews_examples(total_eval_examples)
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_train_examples, f, indent=4)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_eval_examples, f, indent=4)
        if args.debug:
            args.annotation_size = 10
            args.batch_size = 1
            total_train_examples = total_train_examples[:50]
            total_eval_examples = total_eval_examples[:5]
        def format_example(example,label_map,**kwargs):
            prompt = " What is the topic about?\nThe topic is about "
            return f"{example['text']}",\
                   f"{example['label']}", prompt
            # return f"{example['text']} The topic is ",\
            #         f"{label_map[example['label']]}"

        all_train_text_to_encode = ["{}".format(raw_item["text"])
                                    for raw_item in total_train_examples]
        all_eval_text_to_encode = ["{}".format(raw_item["text"])
                                   for raw_item in total_eval_examples]
        label_map = None #{0: "world",1: "sports",2: "business",3: "science"}

    elif task_name=='sst2':
        if False: # os.path.isfile(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) and \
                #os.path.isfile(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')):
            print('use cached examples')
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) as f:
                total_train_examples = json.load(f)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')) as f:
                total_eval_examples = json.load(f)
        else:
            datasets = load_dataset('glue', 'sst2',cache_dir=data_cache_dir)
            total_train_examples = [e for e in datasets['train']]
            total_train_examples = random.sample(total_train_examples, 310)

            total_train_examples = process_sst2_examples(total_train_examples)
            total_eval_examples = [e for e in datasets['validation']]
            total_eval_examples = random.sample(total_eval_examples, 256)
            total_eval_examples = process_sst2_examples(total_eval_examples)
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_train_examples, f, indent=4)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_eval_examples, f, indent=4)
        if args.debug:
            args.annotation_size = 10
            args.batch_size = 1
            total_train_examples = total_train_examples[:50]
            total_eval_examples = total_eval_examples[:5]
        def format_example(example,label_map,**kwargs):
            prompt = "What is my feeling? \n Your feeling is "
            return f"{example['text']} ",\
                   f"{example['label']}", prompt

        all_train_text_to_encode = [raw_item["text"] for raw_item in total_train_examples]
        all_eval_text_to_encode = [raw_item["text"] for raw_item in total_eval_examples]
        #label_map = {0:"negative",1:"positive"}
        label_map=None

    elif task_name=='amazon':
        if False: #os.path.isfile(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) and \
                #os.path.isfile(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')):
            print('use cached examples')
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) as f:
                total_train_examples = json.load(f)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')) as f:
                total_eval_examples = json.load(f)
        else:
            amazon_datasets = load_dataset('amazon_polarity',cache_dir=data_cache_dir)
            total_train_examples = [e for e in amazon_datasets['train']]
            total_train_examples = random.sample(total_train_examples, 310)
            

            total_train_examples = process_amazon_examples(total_train_examples)
            total_eval_examples = [e for e in amazon_datasets['test']]
            total_eval_examples = random.sample(total_eval_examples, 256)
            total_eval_examples = process_amazon_examples(total_eval_examples)
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_train_examples, f, indent=4)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_eval_examples, f, indent=4)
        if args.debug:
            args.annotation_size = 10
            args.batch_size = 1
            total_train_examples = total_train_examples[:50]
            total_eval_examples = total_eval_examples[:5]
        def format_example(example,label_map,**kwargs):
            # return f"Title: {example['title']}\n Review: {example['content']}\n Sentiment:",\
            #        f"{label_map[example['label']]}"
            prompt = "\nMy opinion is "
            return f"{example['text']}",\
                   f"{example['label']}", prompt
        all_train_text_to_encode = [raw_item['text'] for raw_item in total_train_examples]
        all_eval_text_to_encode = [raw_item['text'] for raw_item in total_eval_examples]
        label_map = None

    elif task_name=='ethos':
        if False: #os.path.isfile(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) and \
                #os.path.isfile(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')):
            print('use cached examples')
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) as f:
                total_train_examples = json.load(f)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')) as f:
                total_eval_examples = json.load(f)
        else:
            ethos_datasets = load_dataset('ethos', 'binary', cache_dir=data_cache_dir)
            total_train_examples = [e for e in ethos_datasets['train']]
            all_examples = random.sample(total_train_examples, 310+256)
            total_train_examples = all_examples[:310]
            total_train_examples = process_ethos_examples(total_train_examples)
            total_eval_examples = all_examples[310:]
            total_eval_examples = process_ethos_examples(total_eval_examples)
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_train_examples, f, indent=4)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_eval_examples, f, indent=4)
        if args.debug:
            args.annotation_size = 10
            args.batch_size = 1
            total_train_examples = total_train_examples[:50]
            total_eval_examples = total_eval_examples[:5]
        def format_example(example,label_map,**kwargs):
            prompt = "\nYour speech is "
            return f"Let me hear your speech!\n{example['text']}",\
                   f"{example['label']}", prompt

        all_train_text_to_encode = [raw_item['text'] for raw_item in total_train_examples]
        all_eval_text_to_encode = [raw_item['text'] for raw_item in total_eval_examples]
        label_map = None


    elif task_name=='tweet_hate':
        if False: #os.path.isfile(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) and \
                #os.path.isfile(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')):
            print('use cached examples')
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) as f:
                total_train_examples = json.load(f)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')) as f:
                total_eval_examples = json.load(f)
        else:
            datasets = load_dataset('tweet_eval', 'hate', cache_dir=data_cache_dir)
            total_train_examples = [e for e in datasets['train']]
            total_train_examples = random.sample(total_train_examples, 310)
            total_train_examples = process_tweet_hate_examples(total_train_examples)
            total_eval_examples = [e for e in datasets['test']]
            total_eval_examples = random.sample(total_eval_examples, 256)
            total_eval_examples = process_tweet_hate_examples(total_eval_examples)
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_train_examples, f, indent=4)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_eval_examples, f, indent=4)
        if args.debug:
            args.annotation_size = 10
            args.batch_size = 1
            total_train_examples = total_train_examples[:50]
            total_eval_examples = total_eval_examples[:5]
        def format_example(example,label_map,**kwargs):
            prompt = "\nYour speech is "
            return f"Let me hear your speech!\n{example['text']}",\
                   f"{example['label']}", prompt

        all_train_text_to_encode = [raw_item['text'] for raw_item in total_train_examples]
        all_eval_text_to_encode = [raw_item['text'] for raw_item in total_eval_examples]
        label_map = None

    elif task_name=='complaints':
        if False: # os.path.isfile(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) and \
                #os.path.isfile(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')):
            print('use cached examples')
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) as f:
                total_train_examples = json.load(f)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')) as f:
                total_eval_examples = json.load(f)
        else:
            datasets = load_dataset('ought/raft', 'twitter_complaints',cache_dir=data_cache_dir)
            total_train_examples = [e for e in datasets['train']]
            

            total_train_examples = process_complaints_examples(total_train_examples)
            # total_eval_examples = [e for e in process_complaints_examples['test']]
            # total_eval_examples = random.sample(total_eval_examples, 256)
            total_eval_examples = copy.deepcopy(total_train_examples) #process_sst2_examples(total_eval_examples)
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_train_examples, f, indent=4)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_eval_examples, f, indent=4)
        if args.debug:
            args.annotation_size = 10
            args.batch_size = 1
            total_train_examples = total_train_examples[:50]
            total_eval_examples = total_eval_examples[:5]
        def format_example(example,label_map,**kwargs):
            prompt ="It is "
            return f"Do you have a complaint? \n {example['text']}\n",\
                   f"{example['label']}", prompt
        all_train_text_to_encode = [raw_item["text"] for raw_item in total_train_examples]
        all_eval_text_to_encode = [raw_item["text"] for raw_item in total_eval_examples]
        label_map=None

    else:
        raise ValueError(f"{args.task_name} is not supported")
    return total_train_examples,total_eval_examples,all_train_text_to_encode, \
           all_eval_text_to_encode, format_example,label_map
        


    