import argparse
import json
from argparse import Namespace
from collections.abc import Iterable

import requests


def clear_line(n: int = 1) -> None:
    LINE_UP = "\033[1A"
    LINE_CLEAR = "\x1b[2K"
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


def post_http_request(
    prompt: str, api_url: str, n: int = 1, stream: bool = False
) -> requests.Response:
    headers = {'Content-Type': 'application/json'}
    pload = {
        "model": "Qwen/Qwen3-Reranker-0.6B",
        "query": "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: What is the capital of France?\n",
        "documents": [
    "<Document>: The capital of Brazil is Brasilia.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
     "<Document>: The capital of France is Paris.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
     "<Document>: What is the capital of France?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
     "<Document>: Horses and cows are both animals.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"],
    }
    response = requests.post(api_url, headers=headers, json=pload, stream=stream)
    return response

def get_response(response: requests.Response) -> list[str]:
    data = json.loads(response.content)

    output = data["results"]
    output.sort(key=lambda x: x["index"])
    
    return output

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--stream", action="store_true")
    return parser.parse_args()


def main(args: Namespace):
    prompt = args.prompt
    api_url = f"http://{args.host}:{args.port}/rerank"
    n = args.n

    response = post_http_request(prompt, api_url, n)
    output = get_response(response)
    print([output[i]["relevance_score"] for i in range(len(output))])
if __name__ == "__main__":
    args = parse_args()
    main(args)
    "[9.093089465750381e-05, 0.9996914863586426, 0.990321934223175, 1.1255248864472378e-05]"