import logging

import json
import logging

from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
from typing import Union, List, Tuple, Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data._utils.worker import ManagerWatchdog

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModel, is_torch_npu_available
logger = logging.getLogger(__name__)
#from vllm import LLM, SamplingParams
#from vllm.distributed.parallel_state import destroy_model_parallel
import gc
import math
#from sentence_transformers import CrossEncoder, SentenceTransformer
#from vllm.inputs.data import TokensPrompt
from openai import OpenAI

class Qwen3RerankerOpenAI(torch.nn.Module):
    """
    vllm serve Qwen/Qwen3-Reranker-8B 
    """
    def __init__(self, 
                model_name_or_path,
                instruction="Given the user query, retrieval the relevant passages", 
                api_key: str = "", 
                base_url: str = "http://localhost:8000/v1", 
                **kwargs):
        self.model_name_or_path = model_name_or_path
        self.instruction = instruction
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.suffix = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.max_length=kwargs.get('max_length', 8192)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        
        self.true_token = "yes"
        self.false_token = "no"
        self.true_token_id = self.tokenizer(self.true_token, add_special_tokens=False).input_ids[0]
        self.false_token_id = self.tokenizer(self.false_token, add_special_tokens=False).input_ids[0]

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def format_instruction(self, instruction, query, doc):
        if isinstance(query, tuple):
            instruction = query[0]
            query = query[1]
        text = [
            {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
            {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
        ]
        return text

    def compute_scores(self, pairs, **kwargs):
        messages = [self.format_instruction(self.instruction, query, doc) for query, doc in pairs]
        messages =  self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
        )
        messages = [ele[:self.max_length] + self.suffix_tokens for ele in messages]
        


        response = self.client.completions.create(
            model=self.model_name_or_path,
            prompt=messages,
            top_p=0.95,
            max_tokens=1,
            logprobs=20,
            extra_body={"allowed_token_ids": [self.true_token_id, self.false_token_id]})

        scores = []
        for i in range(len(response.choices)):
            final_logits = response.choices[i].logprobs.top_logprobs[-1]
            
            if self.true_token not in final_logits:
                true_logit = -10
            else:
                true_logit = final_logits[self.true_token]
            if self.false_token not in final_logits:
                false_logit = -10
            else:
                false_logit = final_logits[self.false_token]
            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            score = true_score / (true_score + false_score)
            scores.append(score)

        return scores

    def stop(self):
        pass

if __name__ == '__main__':
    model = Qwen3RerankerOpenAI(model_name_or_path='Qwen/Qwen3-Reranker-8B', instruction="Retrieval document that can answer user's query", max_length=2048)
    queries = ['What is the capital of China?', 'Explain gravity']
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
    ]
    pairs = list(zip(queries, documents))
    new_scores = model.compute_scores(pairs)
    print('scores', new_scores)
    model.stop()
    """
    qwen3-reranker-0.6b
    scores [0.9947798749641705, 0.9982992772280448]

    qwen3-reranker-8b
    scores [0.9959298619216863, 0.9961755163553628]
    """


