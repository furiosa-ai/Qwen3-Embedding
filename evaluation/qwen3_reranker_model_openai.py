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

import gc
import math
from openai import OpenAI

def map_with_progress(
    f: Callable,
    xs: list[Any],
    num_threads: int = os.cpu_count() or 10,
    pbar: bool = True,
):
    """
    Apply f to each element of xs, using a ThreadPool, and show progress.
    """
    pbar_fn = tqdm if pbar else lambda x, *args, **kwargs: x

    if os.getenv("debug"):
        return list(map(f, pbar_fn(xs, total=len(xs))))
    else:
        with ThreadPool(min(num_threads, len(xs))) as pool:
            return list(pbar_fn(pool.imap(f, xs), total=len(xs)))


class Qwen3RerankerInferenceModel(torch.nn.Module):
    """
    vllm serve Qwen/Qwen3-Reranker-0.6B 
    """
    def __init__(self, 
                model_name_or_path,
                instruction="Given the user query, retrieval the relevant passages", 
                api_key: str = "", 
                base_url: str = "http://localhost:8000/v1", 
                **kwargs):

        self.instruction = instruction
        self.model_name_or_path = model_name_or_path
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

    def process_batch(self, pairs, **kwargs):
        # check 
        messages = [self.format_instruction(self.instruction, query, doc) for query, doc, _ in pairs]
        
        messages =  self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
        )
        messages = [ele[:self.max_length] + self.suffix_tokens for ele in messages]
        

        def fn(messages):
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
        
        results = map_with_progress(fn, messages, num_threads=3)
        print(results)
        return results

    def start(self):
        pass

    def predict(
        self,
        sentences: list[tuple[str, str]] | list[list[str]],
        batch_size: int = None,
        show_progress_bar: bool | None = False,
        num_workers: int = 1,
        activation_fct = None,
        apply_softmax: bool | None = False,
        convert_to_numpy: bool =  True,
        convert_to_tensor: bool = False,
        **kwargs
    ) -> list[torch.Tensor]:
        scores = self.process_batch(sentences)
        return scores

    def stop(self):
        pass

