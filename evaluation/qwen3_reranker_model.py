import json
import logging

import torch
from transformers import (
    AutoTokenizer,
)

logger = logging.getLogger(__name__)

import requests
from sentence_transformers import CrossEncoder


class Qwen3RerankerInferenceModel(CrossEncoder):
    def __init__(self, model_name, base_url, instruction="Given the user query, retrieval the relevant passages", **kwargs):

        number_of_gpu=torch.cuda.device_count()
        self.instruction = instruction
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_url = f"{base_url}/score"

        self.query_template = "{prefix}<Instruct>: {instruction}\n\n<Query>: {query}\n\n"
        self.document_template = "<Document>: {doc}{suffix}"

        self.prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n'
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        chat_template_suffix ="<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.chat_template_suffix_token = self.tokenizer.encode(chat_template_suffix, add_special_tokens=False)

        self.max_length=kwargs.get('max_length', 8192)

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
        full_sentences = [self.format_instruction(self.instruction, query, doc) for query, doc in pairs]

        full_sentences =  self.tokenizer.apply_chat_template(
            full_sentences, tokenize=True, add_generation_prompt=False, enable_thinking=False
        )

        full_sentences = [e[:self.max_length] + self.chat_template_suffix_token for e in full_sentences]

        input_docs_truncated = [sentence[q_len:] for sentence, q_len in zip(full_sentences, input_queries_len)]
        input_docs_truncated = [self.tokenizer.decode(ele, skip_special_tokens=True) for ele in input_docs_truncated]

        return input_queries, input_docs_truncated

    def post_http_request(self, queries, documents, stream=False):
        headers = {"Content-Type": "application/json"}
        pload = {
            "model": self.model_name,
            "text_1": queries,
            "text_2": documents,
        }
        response = requests.post(self.base_url,
                                headers=headers,
                                json=pload,
                                stream=stream)

        return response

    def get_score(self, queries, documents):
        response = self.post_http_request(queries, documents)
        outputs = json.loads(response.content)["data"]
        outputs.sort(key=lambda x: x["index"])
        scores = [output["score"] for output in outputs]
        return scores

    def process_batch(self, pairs, **kwargs):
        queries, documents = self.preprocess(pairs)
        scores = self.get_score(queries, documents)

        return scores

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

