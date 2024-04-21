# coding=utf-8
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch OPT model."""
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)

from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.opt.configuration_opt import OPTConfig

from transformers import AutoTokenizer
import transformers


from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP



import multiprocessing
import editdistance
import tqdm

from collections import defaultdict 



from utils import TOKENIZER_MAPS
import os
import pickle
import numpy

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



# MinED mapping between two tokenizers
def find_best_mapping(x, base_tokens, blending_model_special_token, base_model_special_token, best_one=True):
    if x in base_tokens:
        return x , x
    tmp_x = x
    if  x[0] == blending_model_special_token:
        if len(x) == 1 or x[1] != blending_model_special_token:
            tmp_x = x.replace(blending_model_special_token, base_model_special_token, 1)
        
    if tmp_x in base_tokens:
        return x, tmp_x
    else:
        if best_one:
            return x, min([(y, editdistance.eval(tmp_x, y)) for y in base_tokens], key=lambda d: d[1])[0]
        else:
            token_and_distance = [(y, editdistance.eval(tmp_x, y)) for y in base_tokens]
            min_distance = min(item[1] for item in token_and_distance)
            shortest_distance_tokens = [item[0] for item in token_and_distance if item[1] == min_distance]
            return x, shortest_distance_tokens
                    
class PackLLM(nn.Module):

    def __init__(self, fusion, models, model_names, tokenizer=None, tfidf=None, kmeans=None, accelerator=None):
        super(PackLLM, self).__init__()

        self.models = models
        self.model_names = model_names
        assert len(self.models) > 1
       
        self.fusion = fusion
        if self.fusion == 'dexperts' or self.fusion == 'dexperts-greedy':
            assert len(self.model_names) == 3
        self.tokenizer_align_path = "/home/karypisg/mavro016/AdaICL/aligned_tokenizers"

        self.reference_tokenizer = tokenizer

        self.tokenizers = {}
        self.tokenizers_l = []

        use_fast = False
        for tokenizer_name in model_names:
            if  self.reference_tokenizer is None:
                tokenizer_name = TOKENIZER_MAPS[tokenizer_name]
            else:
                tokenizer_name =  self.reference_tokenizer
            if "mosaicml/mpt-7b" in tokenizer_name or "stabilityai/stablelm-3b-4e1t" in tokenizer_name: #GPTNexoXTokenizer has only the fast version.
                use_fast = True
            else:
                use_fast = False
            self.tokenizers[tokenizer_name] = AutoTokenizer.from_pretrained(tokenizer_name, token="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", use_fast=use_fast)
            self.tokenizers_l.append(self.tokenizers[tokenizer_name])
            

        self.tokenizer_list = defaultdict(list) 
        self.vocab_mappings_base = defaultdict(list) 

        self.tfidf = tfidf #cbtm
        self.kmeans = kmeans #cbtm 
        self.accelerator = accelerator
        self.tokenizer_has_map = defaultdict(list) 
        self.aligned_tokenizers = []

        self.current_tokenizer = None
        self.current_expert = None

        self.all_lams = []


    def select_tokenizer(self, test_input):
        """
        Reference tokenizer selection
        """

        tokenized_input = []
        new_inputs = []


        losses = [0] * len(self.models)
        losses_experts = []

        #Output tokenizer: Determine top1 expert
        for (i,tokenizer) in enumerate(self.tokenizers_l):
            expert = self.models[i]
            input_ids = torch.LongTensor(tokenizer(test_input)["input_ids"]).to(expert.device)
            with torch.no_grad():
                outputs = expert.forward(
                    input_ids=input_ids,
                )

            expert_logits = outputs["logits"]
            shift_logits = expert_logits[..., :-1, :].contiguous()

            
            shift_labels = input_ids.to(input_ids.device)
            if tokenizer.eos_token_id is not None:
                ignore_index=tokenizer.eos_token_id
            else:
                ignore_index=0
            # Shift so that tokens < n predict n
            shift_labels = shift_labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(ignore_index=ignore_index)
            loss = loss_fct(shift_logits.view(-1, expert.config.vocab_size), shift_labels.view(-1))
            losses_experts.append(loss.item())

        tokenizer_id = sorted(range(len(losses_experts)), key=lambda i: losses_experts[i])[0] #top1 expert

        return tokenizer_id

    def align_tokenizers(self, tokenizer_id):
        """
        Alignment of tokenizers with reference tokenizer
        """

        if  self.reference_tokenizer is None:
            tokenizer_name = self.model_names[tokenizer_id]
            tokenizer_name = TOKENIZER_MAPS[tokenizer_name] 
        else:
            tokenizer_name =  self.reference_tokenizer
        self.tokenizer = self.tokenizers[tokenizer_name] 
        if tokenizer_name in self.aligned_tokenizers:
            self.current_tokenizer = tokenizer_name
            return
        
        #character-level 
        TOKENIZER_TO_SPECIAL_TOKEN = {transformers.LlamaTokenizer: '▁',
                              transformers.GPTNeoXTokenizerFast: 'Ġ',
                              transformers.CodeGenTokenizer: 'Ġ',
                              transformers.models.codegen.tokenization_codegen.CodeGenTokenizer: 'Ġ',
                              transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer: 'Ġ',
                              transformers.models.gemma.tokenization_gemma.GemmaTokenizer: 'Ġ',
                              transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer: 'Ġ'}

        for name in self.model_names:

            if self.reference_tokenizer is None:
                name = TOKENIZER_MAPS[name] 
            else:
                name =  self.reference_tokenizer

            tokenizer_expert = self.tokenizers[name] 
            self.tokenizer_has_map[tokenizer_name].append(len(self.tokenizer_list[tokenizer_name]))
            self.tokenizer_list[tokenizer_name].append(tokenizer_expert)
            
            d1 = self.tokenizer.get_vocab()
            d2 = tokenizer_expert.get_vocab()
            base_tokens = list(d1.keys())
            blending_tokens = list(d2.keys())
            base_tokenizer = self.tokenizer
            blending_tokenizer = tokenizer_expert
            base_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[base_tokenizer.__class__]
            blending_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[blending_tokenizer.__class__]

            
            file_path = tokenizer_name+"__" + name + "__.pkl"
            file_path = file_path.replace("/", "-")
            file_path = "/home/karypisg/mavro016/AdaICL/aligned_tokenizers/" + file_path
            
            if os.path.isfile(file_path):

                #mapping exists
                with open(file_path, 'rb') as f:
                    base_to_blending_mapping = pickle.load(f)

            else:
                #create new mapping per token
                base_to_blending_mapping = dict()

                with multiprocessing.Pool(4) as pool:
                    mapping_args = [(x, blending_tokens,  base_model_special_token, blending_model_special_token,) for x in
                                    base_tokens]
                    results = list(tqdm.tqdm(pool.starmap(find_best_mapping, mapping_args), total=len(base_tokens)))

                for tmp_x, best_mapping in results:
                    base_to_blending_mapping[d1[tmp_x]] = d2[best_mapping]
                    

                with open(file_path, 'wb') as f:
                    pickle.dump(base_to_blending_mapping, f)

            
            self.vocab_mappings_base[tokenizer_name].append(base_to_blending_mapping)


        self.current_tokenizer = tokenizer_name
        self.aligned_tokenizers.append(tokenizer_name)


    def map_tokens(self, model_id, input_ids):
        """
        Map tokens for inference
        """
        if self.tokenizer_list[self.current_tokenizer][model_id] != self.tokenizers[self.current_tokenizer]: #different tokenizers
            mapping = self.vocab_mappings_base[self.current_tokenizer][model_id] #get mapping
            
            input_ids_list = input_ids[0].tolist()
            new_input_ids = torch.LongTensor([mapping[x] for x in input_ids_list]).unsqueeze(0).to(input_ids.device)
            return new_input_ids
        else:
            return input_ids
            

    def set_expert(self, model_id):
        self.current_expert = self.models[model_id]

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def _get_context_clusters_seq(self, net_input, tokenizer):
        """
        cBTM clustering and expert fusion
        """
        
        temperature = 0.1
        
        prompt = net_input[0]
        #cbtm embeddings
        decoded_tokens = tokenizer.batch_decode(prompt.unsqueeze(0))
        _, distances = self.kmeans.predict(torch.from_numpy(self.tfidf.transform(decoded_tokens)),
                                    return_distances=True)

        if distances.size(-1) == len(self.models) - 1: #large base model 
            mean_dist = torch.mean(distances, dim=-1, keepdim=True)
            distances = torch.cat((distances, mean_dist), dim=-1)
        
        cluster_assignments = torch.nn.functional.softmax(-distances**2 / temperature, dim=-1)
        

        return cluster_assignments.tolist()



    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForCausalLM

        >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious. I'm just a little bit of a weirdo."
        ```"""

        
        input_ids = input_ids.to(self.accelerator.device)
        attention_mask = attention_mask.to(self.accelerator.device)

        #labels for peplexity
        shift_labels = input_ids.to(input_ids.device)
        if self.tokenizers[self.current_tokenizer].eos_token_id is not None:
            ignore_index=self.tokenizers[self.current_tokenizer].eos_token_id
        else:
            ignore_index=0
        # Shift so that tokens < n predict n
        shift_labels = shift_labels[..., 1:].contiguous()
        

        expert_losses = []
        logits_all = []
        loss_fct = CrossEntropyLoss(ignore_index=ignore_index)
        #inference for ppl estimation
        for (model_id, expert) in enumerate(self.models ):
            with torch.no_grad():
                #per tokenizer input
                new_input_ids = self.map_tokens(model_id, input_ids.clone())

                outputs = expert.forward(
                    input_ids=new_input_ids,
                    attention_mask=None,  #work for all models
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                expert_logits = outputs["logits"]
                shift_logits = expert_logits[..., :-1, :].contiguous()

                #mapping back to reference tokenizer
                if self.tokenizer_list[self.current_tokenizer][model_id] != self.tokenizers[self.current_tokenizer]: #different tokenizers
                    base_vocab_ids = torch.LongTensor(list(self.vocab_mappings_base[self.current_tokenizer][model_id].values())).to(input_ids.device)
                    shift_logits = shift_logits[:,:,base_vocab_ids]
                if expert_logits.size(-1) > len(self.tokenizers[self.current_tokenizer]):  # for medalpaca, phi, etc.
                    shift_logits = shift_logits[:,:,:len(self.tokenizers[self.current_tokenizer])].contiguous()
                logits_all.append(shift_logits)
                
                #perplexiry computation
                loss = loss_fct(shift_logits.view(-1, len(self.tokenizers[self.current_tokenizer])), shift_labels.view(-1))
                expert_losses.append(loss)
        expert_losses = torch.FloatTensor(expert_losses).to(input_ids.device)
        
        if self.fusion=='opt':
            sorted_idx = torch.argsort(expert_losses)
            #initialize lambdas
            lams = [0] *sorted_idx.size(-1)
            lams[0] = 1
            #start from top1 expert
            curr_logits = logits_all[sorted_idx[0]]

            upper_lamb=21 #step 0.05
            for step in range(1, len(sorted_idx)):
                
                best_loss = 10000
                #next expert
                next_logits = logits_all[sorted_idx[step]]
                best_lamb=1
                #grid search
                for lamb in range(0,upper_lamb): #lamb = 0 to 1, step 0.05
                    lam = lamb / (upper_lamb-1)
                    with torch.no_grad():
                        expert_logits = lam * curr_logits + (1-lam) * next_logits
                        loss = loss_fct(expert_logits.view(-1, expert_logits.size(-1)), shift_labels.view(-1))
                        if loss < best_loss:
                            best_loss = loss.item()
                            best_lamb = lam
                curr_logits = best_lamb * curr_logits + (1-best_lamb) * next_logits #new logits

                #if best_lamb == 1: break #early stop
                for lam_id in range(step):
                    lams[lam_id] *= best_lamb
                lams[step] += 1-best_lamb  #gather all lambdas

            self.all_lams.append(lams)
            print("sorted experts: ", sorted_idx.tolist())
            print("test lambdas sorted: ", lams)
            #print("all lambdas: ", numpy.mean(self.all_lams, axis=0) ) 

        elif self.fusion== 'sim':
            tau = 1
            lams = nn.functional.softmax(-expert_losses/tau).tolist()
            curr_logits = 0
            for i,logit in enumerate(logits_all):
                curr_logits += lams[i] * logit
            self.all_lams.append(sorted(lams, reverse=True))
            print("test lambdas unsorted: ", lams)
            #print("lambdas: ", numpy.mean(self.all_lams, axis=0) )

        elif self.fusion=='ensemble':
            curr_logits = 0
            for logit in logits_all:
                curr_logits += logit
            curr_logits = curr_logits / len(logits_all)
        elif self.fusion=='top1':
            sorted_idx = torch.argsort(expert_losses)
            curr_logits = logits_all[sorted_idx[0]] #single expert
            print("top1 id: ", sorted_idx[0])

        elif self.fusion=='cbtm':
            lams = self._get_context_clusters_seq(input_ids,  self.tokenizers[self.current_tokenizer])
            curr_logits = 0
            for i,logit in enumerate(logits_all):
                curr_logits += lams[i] * logit
            print("test lambdas unsorted: ", lams)


        elif self.fusion=='dexperts':
            curr_logits = 0
            alpha = 1 
            curr_logits += alpha*logits_all[0]
            curr_logits -= alpha*logits_all[1]
            curr_logits += logits_all[2]

        elif self.fusion=='dexperts-opt':
            base_logits = logits_all[2]

            best_loss = 10000
            upper_lamb=201
            for lamb in range(0,upper_lamb): #lamb = 0 to 1, step 0.005
                lam = lamb / (upper_lamb-1)

                with torch.no_grad():
                    expert_logits = base_logits + lam * (logits_all[0] - logits_all[1])
                    loss = loss_fct(expert_logits.view(-1, expert_logits.size(-1)), shift_labels.view(-1))
                    
                    if loss < best_loss:
                        best_loss = loss.item()
                        best_lamb = lam
                        curr_logits = expert_logits

            print("Best lambda dexpert: ", best_lamb)

        out_logits = curr_logits
        loss = None

        #print("loss", loss, logits.device)
        if not return_dict:
            output = (out_logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=out_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past