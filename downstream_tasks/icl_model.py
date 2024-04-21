# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
import torch
import torch.nn.functional as F

from tqdm import tqdm
from transformers import Adafactor, AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelForCausalLM, AutoConfig, LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from modeling_llm_fusion import PackLLM


class MetaICLModel(object):

    def __init__(self, accelerator=None, logger=None, out_dir=None, fp16=True, local_rank=-1,args=None):
        if logger is None:
            class Logger():
                def info(self, text):
                    print ("Logging from MetaICLModel:\t", text)
            logger = Logger()

        self.logger = logger
        self.out_dir = out_dir
        self.fp16 = fp16
        self.local_rank = local_rank
        self.args = args

        self.accelerator = accelerator
        if self.local_rank == -1:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            n_gpu = torch.cuda.device_count()
            ws = 1
        else:  # distributed mode
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            ws = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
            torch.distributed.init_process_group(backend="nccl")
            n_gpu = 1

        self.n_gpu = n_gpu
        self.device = device
        if self.local_rank <= 0:
            logger.info("Setting up for local_rank=%d, world_size=%d" % (self.local_rank, ws))
        self.model_name = None
        self.model = None
        self.mode = None

    def __str__(self):
        text = "[MetaICL Model]: "
        if self.model_name is None:
            text += "No model loaded yet"
        else:
            text += self.model_name
            if self.mode is None:
                text += " (no mode setted - try .train() or .eval()"
            else:
                text += " (%s mode)" % self.mode
        text += "\nusing device %s, %d gpus, local_rank=%d" % (self.device, self.n_gpu, self.local_rank)
        return ("="*50) + "\n" + text + "\n" + ("="*50)

    def is_none(self):
        return self.model is None

    def train(self):
        self.model.train()
        self.mode = "train"

    def eval(self):
        self.model.eval()
        self.mode = "eval"

    def cuda(self):
        self.model.cuda()
        for i, model in enumerate(self.model.models):
            self.model.models[i] = model.cuda()

    def to_device(self):
        self.model.to(self.device)
        for i, model in enumerate(self.model.models):
            self.model.models[i].to(self.device)

    def load(self, fusion, tokenizer):

        models = []
        model_path_list = self.args.model_name.split(",")
        for model_path in model_path_list:
            if 'llama' in model_path:
                #check your token
                config = AutoConfig.from_pretrained(model_path, token="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")
                model = LlamaForCausalLM.from_pretrained(
                    model_path,
                    from_tf=bool(".ckpt" in model_path),
                    use_flash_attention_2=False,
                    token="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",
                    config=config,
                    torch_dtype=torch.float16,
                    #device_map={"": self.accelerator.process_index},
                    device_map="balanced" if torch.cuda.device_count() > 1 else "auto"
                )
                model = model.half().eval()
            else:
                config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    from_tf=bool(".ckpt" in model_path),
                    use_flash_attention_2=False,
                    torch_dtype=torch.float16,
                    token="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",
                    config=config,
                    trust_remote_code=True,
                    device_map="balanced" if torch.cuda.device_count() > 1 else "auto",
                    
                )
                model = model.half().eval()

            models.append(model)
        print(models)
        
        model = PackLLM(fusion, models, model_path_list, tokenizer=tokenizer)
        self.model = model
        self.model_name = model_path_list[-1]

    def save(self, step):
        if self.local_rank <= 0:
            model_state_dict = {key[7:] if key.startswith("module.") else key: value.cpu()
                                for key, value in self.model.state_dict().items()}
            torch.save(model_state_dict, os.path.join(self.out_dir, "model-{}.pt".format(step)))
            self.logger.info("Saving model parameters at step=%d" % step)

    def do_inference(self, data, batch_size=1, verbose=False):
        dataloader = data.get_dataloader(batch_size, is_training=False)
        #dataloader = self.accelerator.prepare(dataloader)
        if verbose:
            dataloader = tqdm(dataloader)
        losses = []
        for batch in dataloader:
            input_ids=batch[0].to(self.accelerator.device) #.cuda()
            attention_mask=batch[1].to(self.accelerator.device) #.cuda()
            token_type_ids=batch[2].to(self.accelerator.device) #.cuda()
            test_input=batch[3].to(self.accelerator.device) #.cuda()
            prompt=batch[4].to(self.accelerator.device) #.cuda()
            if len(batch)==5:
                labels=None
            else:
                labels=batch[5].to(self.accelerator.device) #.cuda()
            with torch.no_grad():
                loss = self.run_model(input_ids, attention_mask, token_type_ids, labels=labels)
            
            losses += loss.cpu().detach().numpy().tolist()
            
        return losses

    def do_predict(self, data, batch_size=1, losses=None, verbose=False, require_loss=False, label_id=None):
        

        if losses is None:
            losses = self.do_inference(data, batch_size, verbose=verbose)
        
        losses = np.array(losses)
        assert len(losses) == len(data)
        predictions = []
        for idx, dp in enumerate(data.metadata):
            curr_label_losses = [np.sum(losses[indices]) for indices in dp["indices"]]
            if label_id is not None:
                prediction = dp["options"][label_id]
                negative_pred_prob = curr_label_losses[label_id]
                predictions.append([prediction.strip(), negative_pred_prob])
            if not require_loss:
                prediction_idx = sorted(enumerate(curr_label_losses), key=lambda x: x[1])[0][0]
                prediction = dp["options"][prediction_idx]
                predictions.append(prediction.strip())
            else:
                prediction_terms = sorted(enumerate(curr_label_losses), key=lambda x: x[1], reverse=False)[0]
                prediction = dp["options"][prediction_terms[0]]
                negative_pred_prob = prediction_terms[1]
                predictions.append([prediction.strip(), negative_pred_prob])
                

        return predictions
    
    

    def run_model(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
        if labels is None:
            labels = input_ids
        labels = labels[..., 1:].contiguous()
    

        #label mask filters out non-labels
        label_mask = token_type_ids[..., 1:].contiguous()

        #"""
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)) # [batch_size, length]

        losses = losses.view(logits.size(0), logits.size(1)) * label_mask
        norm_losses = torch.sum(losses, axis=1) / torch.sum(label_mask, axis=1)
        
        """
        norm_losses = compute_loss(logit, labels, label_mask)
        norm_losses = norm_losses.unsqueeze(0).to(logit.device)
        """
        labels = None


        return norm_losses



def setup_fp16(model, optimizer):
    try:
        import apex
        from apex import amp
        apex.amp.register_half_function(torch, "einsum")
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    fp16_opt_level = "O1"
    model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)
    return model, optimizer



