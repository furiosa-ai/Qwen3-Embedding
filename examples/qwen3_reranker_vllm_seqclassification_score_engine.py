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
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
import gc
import math
from sentence_transformers import CrossEncoder, SentenceTransformer
from vllm.inputs.data import TokensPrompt


class Qwen3RerankervllmSeqClass(CrossEncoder):
    """
    https://docs.vllm.ai/en/v0.11.2/examples/offline_inference/pooling/
    
    """
    def __init__(self, model_name_or_path, instruction="Given the user query, retrieval the relevant passages", **kwargs):
        number_of_gpu=torch.cuda.device_count()

        self.instruction = instruction
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token


        self.query_template = "{prefix}<Instruct>: {instruction}\n\n<Query>: {query}\n\n"
        self.document_template = "<Document>: {doc}{suffix}"

        self.prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n'
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        chat_template_suffix ="<|im_start|>assistant\n<think>\n\n</think>\n\n" 
        self.chat_template_suffix_token = self.tokenizer.encode(chat_template_suffix, add_special_tokens=False)

        self.max_length=kwargs.get('max_length', 8192)
        
        self.lm = LLM(
            model=model_name_or_path,
            runner="pooling",
            hf_overrides={
                "architectures": ["Qwen3ForSequenceClassification"],
                "classifier_from_token": ["no", "yes"],
                "is_original_qwen3_reranker": True,
            },
        )

    def format_instruction(self, instruction, query, doc):
        if isinstance(query, tuple):
            instruction = query[0]
            query = query[1]
        text = [
            {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
            {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
        ]

        return text

    def preprocess(self, pairs):
        input_queries = [self.query_template.format(prefix=self.prefix, instruction=self.instruction, query=query) for query, doc in pairs] 
        input_queries_len = [len(self.tokenizer.encode(query, add_special_tokens=False)) for query in input_queries]
        
        # truncate documents if needed 
        full_sentences = [self.format_instruction(instruction, query, doc) for query, doc in pairs]

        full_sentences =  self.tokenizer.apply_chat_template(
            full_sentences, tokenize=True, add_generation_prompt=False, enable_thinking=False
        )

        full_sentences = [e[:self.max_length] + self.chat_template_suffix_token for e in full_sentences]
        print("input_queries_len[0]", input_queries_len[0])
        input_docs_truncated = [sentence[q_len:] for sentence, q_len in zip(full_sentences, input_queries_len)]
        input_docs_truncated = [self.tokenizer.decode(ele, skip_special_tokens=True) for ele in input_docs_truncated]
        return input_queries, input_docs_truncated

    def compute_scores(self, pairs, **kwargs):
        #TODO check if TokensPrompt can be applied.
        queries, documents = self.preprocess(pairs)
        outputs = self.lm.score(queries, documents)
        scores = [output.outputs.score for output in outputs]
        
        return scores

    def stop(self):
        pass

if __name__ == '__main__':
    instruction = "Given a web search query, retrieve relevant passages that answer the query"
    model = "Qwen/Qwen3-Reranker-8B"
    model = Qwen3RerankervllmSeqClass(model_name_or_path=model, instruction=instruction, max_length=2048)
    queries = [
        "What is the capital of France?"
    ]

    documents = [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris.",
        "What is the capital of France?",
        "Horses and cows are both animals.",
    ]
    pairs = list(zip(queries*len(documents), documents))
    new_scores = model.compute_scores(pairs)
    print('scores', new_scores)
    model.stop()
    """
    qwen3-reranker-0.6b
    scores [0.00018043820455204695, 0.9932016730308533, 0.8553178906440735, 2.733712608460337e-05]

    qwen3-reranker-8b
    scores [5.066717858426273e-05, 0.984429121017456, 0.48676350712776184, 7.622045814059675e-06]
    """



