#coding:utf8
from typing import Dict, Optional, List, Union
import torch
import vllm
from vllm import LLM, PoolingParams
from vllm.distributed.parallel_state import destroy_model_parallel

class Qwen3EmbeddingVllm():
    def __init__(self, model_name_or_path, instruction=None, max_length=8192):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        self.instruction = instruction
        self.model = LLM(model=model_name_or_path, task="embed", hf_overrides={"is_matryoshka": True})

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
            output = self.model.embed(sentences,pooling_params=PoolingParams(dimensions=dim))
        else:
            output = self.model.embed(sentences)
        output = torch.tensor([o.outputs.embedding for o in output])
        return output


    def stop(self):
        destroy_model_parallel()

if __name__ == "__main__":
    model_path = "Qwen/Qwen3-Embedding-8B"
    model = Qwen3EmbeddingVllm(model_path)
    queries = ['What is the capital of China?', 'Explain gravity']
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
    ]
    query_outputs = model.encode(queries, is_query=True)
    doc_outputs = model.encode(documents)
    print('query outputs', query_outputs)
    print('doc outputs', doc_outputs)
    scores = (query_outputs @ doc_outputs.T) * 100
    print(scores.tolist())
    model.stop()

    """
    qwen3-embedding-0.6b
    query outputs tensor([[-0.0507, -0.0303, -0.0002,  ...,  0.0741,  0.0357, -0.0122],
        [-0.0109, -0.0339, -0.0011,  ..., -0.0262,  0.0024, -0.0022]])
    doc outputs tensor([[-0.0470, -0.0218,  0.0036,  ...,  0.0554,  0.0709, -0.0175],
            [-0.0535, -0.0159, -0.0013,  ...,  0.0039, -0.0209,  0.0199]])
    [[76.28856658935547, 13.991527557373047], [13.373743057250977, 59.88606262207031]]

    qwen3-embedding-8b 
    query outputs tensor([[-0.0267,  0.0470, -0.0160,  ..., -0.0112,  0.0044,  0.0151],
        [ 0.0331, -0.0070, -0.0067,  ..., -0.0059, -0.0097,  0.0075]])
    doc outputs tensor([[-0.0136,  0.0521, -0.0033,  ..., -0.0042,  0.0094,  0.0111],
            [ 0.0391,  0.0194, -0.0110,  ..., -0.0096,  0.0064, -0.0007]])
    [[74.88276672363281, 7.471127986907959], [8.884208679199219, 63.24579620361328]]
    """
