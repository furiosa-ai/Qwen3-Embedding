#coding:utf8
from typing import Dict, Optional, List, Union
import torch
import vllm
from vllm import LLM, PoolingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from openai import OpenAI

class Qwen3EmbeddingOpenAI():
    """
    vllm serve Qwen/Qwen3-Embedding-8B
    """
    def __init__(self, 
                model_name_or_path, 
                instruction=None,
                api_key: str = "", 
                base_url: str = "http://localhost:8000/v1", 
                max_length=8192):
        self.model_name_or_path = model_name_or_path
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        self.instruction = instruction
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        if task_description is None:
            task_description = self.instruction
        return f'Instruct: {task_description}\nQuery:{query}'

    def encode(self, sentences: Union[List[str], str], is_query: bool = False, instruction=None, dim: int = -1):
        if isinstance(sentences, str):
            sentences = [sentences]
        if is_query:
            sentences = [self.get_detailed_instruct(instruction, sent) for sent in sentences]
        if dim > 0:
            output = self.client.embeddings.create(model=self.model_name_or_path, 
                                        input = sentences,
                                        encoding_format="float",
                                        extra_body={
                                            "dimensions": dim},
                                        )
            """
            'Model "Qwen/Qwen3-Embedding-0.6B" does not support matryoshka representation, 
            changing output dimensions will lead to poor results.'
            'Model "Qwen/Qwen3-Embedding-8B" does not support matryoshka representation,
             changing output dimensions will lead to poor results.'
            """
        else:
            output = self.client.embeddings.create(model=self.model_name_or_path, 
                                        input = sentences,
                                        encoding_format="float")
        output = torch.tensor([o.embedding for o in output.data])
        return output


    def stop(self):
        pass

if __name__ == "__main__":
    model_path = "Qwen/Qwen3-Embedding-8B"
    model = Qwen3EmbeddingOpenAI(model_path)
    queries = ['What is the capital of China?', 'Explain gravity']
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
    ]

    query_outputs = model.encode(queries, is_query=True, dim=1024)
    doc_outputs = model.encode(documents)
    print('query outputs', query_outputs)
    print('doc outputs', doc_outputs)
    scores = (query_outputs @ doc_outputs.T) * 100
    print(scores.tolist())
    model.stop()
    """
    qwen3-embedding-0.6b 
    query outputs tensor([[-0.0509, -0.0298, -0.0002,  ...,  0.0735,  0.0357, -0.0121],
        [-0.0110, -0.0326, -0.0006,  ..., -0.0263,  0.0026, -0.0022]])
    doc outputs tensor([[-0.0468, -0.0210,  0.0038,  ...,  0.0563,  0.0705, -0.0175],
            [-0.0532, -0.0157, -0.0013,  ...,  0.0037, -0.0211,  0.0203]])
    [[76.38825225830078, 14.075353622436523], [13.473434448242188, 60.02253341674805]]

    qwen3-embedding-8b
    query outputs tensor([[-0.0267,  0.0470, -0.0160,  ..., -0.0112,  0.0044,  0.0151],
        [ 0.0331, -0.0070, -0.0067,  ..., -0.0059, -0.0097,  0.0075]])
    doc outputs tensor([[-0.0139,  0.0526, -0.0033,  ..., -0.0043,  0.0096,  0.0113],
            [ 0.0391,  0.0194, -0.0110,  ..., -0.0096,  0.0064, -0.0007]])
    [[74.84983825683594, 7.471127986907959], [8.81698226928711, 63.24579620361328]]
    """
