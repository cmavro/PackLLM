# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import csv
import json
import string
import numpy as np
import pickle as pkl
import math
import torch

from collections import defaultdict
from functools import partial
from multiprocessing import Pool

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer

from utils import TOKENIZER_MAPS



class MetaICLData(object):

    def __init__(self, logger=None, tokenizer_name=None, model_name=None, method="channel", use_demonstrations=True, k=16,
                 max_length=1024, max_length_per_example=256,
                 do_tensorize=False, tensorize_dir=None, n_process=None, n_gpu=None, local_rank=-1):

        self.logger = logger
        #self.tokenizer = tokenizer
        self.method = method
        self.use_demonstrations = use_demonstrations
        self.k = k
        self.max_length = max_length
        self.max_length_per_example = max_length_per_example

        self.do_tensorize = do_tensorize
        self.tensorize_dir = tensorize_dir
        self.n_process = n_process
        self.n_gpu = n_gpu
        self.local_rank = local_rank

        self.tensorized_inputs = None
        self.metadata = None
        
        
        if tokenizer_name is None:
            tokenizer_name = TOKENIZER_MAPS[model_name]

        if "mosaicml/mpt-7b" in tokenizer_name or "stabilityai/stablelm-3b-4e1t" in tokenizer_name: #GPTNexoXTokenizer has only the fast version.
            use_fast = True
        else:
            use_fast = False
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token="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", use_fast=use_fast)



    def __len__(self):
        if self.tensorized_inputs is None:
            return 0
        return len(self.tensorized_inputs["input_ids"])

    def __str__(self):
        text = "[MetaICL Data]: method=%d, "
        if self.use_demonstrations:
            text += "%d demonstrations\n" % self.k
        else:
            text += "no demonstrations\n"
        if self.metadata is None:
            text += "Currently not containing any examples"
        else:
            text += "Currently containing %d examples with %d tensors to be fed in\n" % (len(self.metadata), len(self))
            text += "\n"
            text += self.print_tensorized_example(return_string=True)
        return ("="*50) + "\n" + text + "\n" + ("="*50)

    def get_dataloader(self, batch_size, is_training, return_dataset=False):
        inputs = self.tensorized_inputs
        for k, v in inputs.items():
            if type(v)==list:
                inputs[k] = torch.LongTensor(v)
        shape = inputs["input_ids"].shape

        test_input = inputs.pop("test_input")
        prompt = inputs.pop("prompt")
        # self.logger.info(shape)
        for v in inputs.values():
            assert (v.shape==shape)
        inputs["test_input"] = test_input
        inputs["prompt"] = prompt
        if "labels" in inputs:
            dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"], inputs["test_input"], inputs["prompt"], inputs["labels"])
        else:
            dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"], inputs["test_input"], inputs["prompt"])
        if is_training:
            sampler=RandomSampler(dataset)
        else:
            sampler=SequentialSampler(dataset)
        if return_dataset:
            return dataset
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        return dataloader

    def _prepro_each_datapoint(self, dp, is_first=True, is_training=False, for_demonstrations=False,
                               add_newlines=True):
        dp = dp.copy()
        if add_newlines:
            no_label = np.all([option=="" for option in dp["options"]])
            no_input = dp["input"]==""
            if self.method=="direct":
                if not is_first:
                    if no_input:
                        dp["input"] = "\n\n" + dp["input"]
                    else:
                        dp["input"] = "\n\n\n" + dp["input"]
                if not no_label:
                    dp["output"] = "\n" + dp["output"]
                    if "options" in dp:
                        dp["options"] = ["\n" + opt for opt in dp["options"]]
            elif self.method=="channel":
                if not is_first:
                    dp["output"] = "\n\n\n" + dp["output"]
                    if "options" in dp:
                        dp["options"] = ["\n\n\n" + opt for opt in dp["options"]]
                if not no_input:
                    if not no_label:
                        dp["input"] = "\n" + dp["input"]
            else:
                raise NotImplementedError()
        else:
            if not is_first:
                if self.method=="direct":
                    dp["input"] = " " + dp["input"]
                elif self.method=="channel":
                    dp["output"] = " " + dp["output"]
                    if "options" in dp:
                        dp["options"] = [" "+opt for opt in dp["options"]]
                else:
                    raise NotImplementedError()
            if self.method=="direct":
                dp["output"] = " " + dp["output"]
                if "options" in dp:
                    dp["options"] = [" " + opt for opt in dp["options"]]
            elif self.method=="channel":
                dp["input"] = " " + dp["input"]
            else:
                raise NotImplementedError()

        input_tokens = self.tokenizer(dp["input"])["input_ids"]

        if is_training or for_demonstrations:
            output_tokens = self.tokenizer(dp["output"],  add_special_tokens=False)["input_ids"]

            if "task" in dp:
                if (dp["task"].startswith("inst:piqa") or dp["task"].startswith("inst:yahoo_answers_topics")) and \
                        len(input_tokens)+len(output_tokens)+2>self.max_length_per_example:
                    input_tokens = input_tokens[:self.max_length_per_example // 2]
                    output_tokens = output_tokens[:self.max_length_per_example // 2 - 2]

                elif len(input_tokens)>=self.max_length_per_example - 2 - len(output_tokens):
                    if dp["task"].startswith("inst:") and len(input_tokens)<len(output_tokens):
                        output_tokens = output_tokens[:self.max_length_per_example - 2 - len(input_tokens)]
                    else:
                        input_tokens = input_tokens[:self.max_length_per_example - 2 - len(output_tokens)]

            assert len(input_tokens)+len(output_tokens)+2<=self.max_length_per_example, \
                (dp.get("task", None), len(input_tokens), len(output_tokens), self.max_length_per_example)

            if self.method=="direct":
                return input_tokens, output_tokens
            elif self.method=="channel":
                return output_tokens, input_tokens
            else:
                raise NotImplementedError()

        else:
            assert len(dp["options"])>=2, dp
            assert dp["output"] in dp["options"]
            option_tokens = [self.tokenizer(option,  add_special_tokens=False)["input_ids"] for option in dp["options"]]
            option_length = np.max([len(option) for option in option_tokens])

            if len(input_tokens)>=self.max_length_per_example - 2 - option_length:
                input_tokens = input_tokens[:self.max_length_per_example - 2 - option_length]

            input_tokens = [input_tokens for _ in option_tokens]
            output_tokens = option_tokens
            option_tokens = [dp["options"].index(dp["output"])]

            if self.method=="direct":
                return input_tokens, output_tokens, option_tokens
            elif self.method=="channel":
                return output_tokens, input_tokens, option_tokens
            else:
                raise NotImplementedError()

    def tensorize(self, _train_data, _test_data, options=None,
                  add_newlines=True, return_list=False, prompt=None):
        if options is not None:
            assert np.all([dp["output"] in options for dp in _train_data])
            for i, dp in enumerate(_test_data):
                # assert "options" not in dp
                assert type(dp)==str
                _test_data[i] = {"input": dp, "options": options}

        train_data, test_data = [], []
        if self.use_demonstrations:
            for dp in _train_data:
                assert type(dp)==dict, ("Each example should be a dictionary", dp)
                assert "input" in dp and "output" in dp, ("Training example should contain input and output", dp)
                train_data.append(dp.copy())
        for dp in _test_data:
            assert type(dp)==dict, ("Each example should be a dictionary", dp)
            assert "input" in dp and "options" in dp and type(dp["options"])==list, \
                ("Test example should contain input and options in a list format", dp)
            if "output" not in dp:
                dp["output"] = dp["options"][0] # randomly choose one (we don't need it anyways)
            test_data.append(dp.copy())

        # each datapoint: passage, question, options, output
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        input_ids, attention_mask, token_type_ids = [], [], []
        prompts = []
        test_input = []

        metadata = []

        if self.use_demonstrations:
            assert len(train_data)==self.k
            demonstrations = []
            for i, dp in enumerate(train_data):
                input_, output_ = self._prepro_each_datapoint(
                    dp, is_first=i==0, for_demonstrations=True,
                    add_newlines=add_newlines)
                demonstrations += input_ + output_

        if prompt is not None:
            prompt = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            prompt = prompt[0]

        cur_len = 0
        for dp_idx, dp in enumerate(test_data):

            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)

            indices = [[i] for i in range(len(input_ids), len(input_ids)+len(inputs))]

            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})

            for inputs_, outputs_ in zip(inputs, outputs):

                
                if self.use_demonstrations:
                    inputs_ = demonstrations + inputs_
                if prompt is not None:
                    inputs_ = inputs_ + prompt
                
                cur_len = max(cur_len, len(inputs_)+len(outputs_))
            for inputs_, outputs_ in zip(inputs, outputs):
                if prompt is not None:
                    prompts.append(prompt)
                else:
                    prompts.append(inputs_)
                test_input.append(inputs_ + prompt)
                if self.use_demonstrations:
                    inputs_ = demonstrations + inputs_
                if prompt is not None:
                    inputs_ = inputs_ + prompt
                    
                encoded = prepro_sentence_pair_single(
                    inputs_, outputs_, cur_len, bos_token_id, eos_token_id,
                    allow_truncation=self.use_demonstrations)

                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])

        if return_list:
            return (input_ids, attention_mask, token_type_ids, test_input, prompts, metadata)

       
        self.tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids),
                                    attention_mask=torch.LongTensor(attention_mask),
                                    token_type_ids=torch.LongTensor(token_type_ids),
                                    test_input=torch.LongTensor(test_input),
                                    prompt=torch.LongTensor(prompts))
        self.metadata = metadata


    def print_tensorized_example(self, return_string=False):
        assert self.tensorized_inputs is not None

        idx = 0
        text = "Checking the first example..."
        input_ids = self.tensorized_inputs["input_ids"][idx]
        token_type_ids = self.tensorized_inputs["token_type_ids"][idx]
        if type(input_ids)!=list:
            input_ids = input_ids.numpy().tolist()
        if type(token_type_ids)!=list:
            token_type_ids = token_type_ids.numpy().tolist()

        text += "\nInput:\n"
        text += self.tokenizer.decode(input_ids[:token_type_ids.index(1)])
        text += "\nOutput:\n"
        text += self.tokenizer.decode([_id for _id, _type_id in zip(input_ids, token_type_ids) if _type_id==1])

        if return_string:
            return text

def prepro_sentence_pair_single(ids1, ids2, max_length,
                                bos_token_id, eos_token_id,
                                allow_truncation=False):

    #if bos_token_id is not None:
    #    ids1 = [bos_token_id] + ids1
    #if eos_token_id is not None:
    #    ids2 = ids2 + [eos_token_id]
    if allow_truncation and len(ids1)+len(ids2) > max_length:
        ids1 = ids1[len(ids1)+len(ids2)-max_length:] # len = max_length-len(ids2)
        assert len(ids1)+len(ids2)==max_length

    n_mask = min(max_length-len(ids1)-len(ids2), max_length-len(ids1)-len(ids2))
    assert n_mask>=0, (max_length, len(ids1), len(ids2))
    if eos_token_id is not None:

        input_ids = ids1+ids2+[eos_token_id for _ in range(n_mask)]
    else:
        input_ids = ids1+ids2+[0 for _ in range(n_mask)]
    attention_mask = [1 for _ in ids1+ids2] + [0 for _ in range(n_mask)]
    token_type_ids = [0 for _ in ids1] + [1 for _ in ids2] + [0 for _ in range(n_mask)]

    #print("inp", len(input_ids))
    return input_ids, attention_mask, token_type_ids

def prepro_sentence_pair(train_inputs, test_inputs, max_length,
                         bos_token_id, eos_token_id,
                         allow_truncation=False):
    input_ids, attention_mask, token_type_ids = [], [], []
    for test_input in test_inputs:
        for train_input in train_inputs:
            _input_ids, _attention_mask, _token_type_ids = \
                prepro_sentence_pair_single(train_input, test_input, max_length,
                                            bos_token_id, eos_token_id,
                                            allow_truncation=allow_truncation)
            input_ids.append(_input_ids)
            attention_mask.append(_attention_mask)
            token_type_ids.append(_token_type_ids)

    return {"input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(attention_mask),
            "token_type_ids": torch.LongTensor(token_type_ids)}

