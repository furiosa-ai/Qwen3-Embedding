import json
from transformers import AutoTokenizer
import requests

url = "http://127.0.0.1:8000/v1"

headers = {"accept": "application/json", "Content-Type": "application/json"}
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

true_token = "yes"
false_token = "no"

true_token_id = tokenizer(true_token, add_special_tokens=False).input_ids[0]
false_token_id = tokenizer(false_token, add_special_tokens=False).input_ids[0]

"""
vllm serve Qwen/Qwen3-Reranker-0.6B 
"""

def main():
    import openai

    instruction = "Given the user query, retrieval the relevant passages"
    query = "What is the capital of France?"
    doc = "The capital of France is Paris."

    text = [
            {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
            {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
        ],

    text =  tokenizer.apply_chat_template(
            text, tokenize=True, add_generation_prompt=False, enable_thinking=False
        )
    
    max_length = 100
    suffix = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    messages = [t[:max_length] + suffix_tokens for t in text] 
    #print(messages)

    client = openai.OpenAI(api_key="", base_url=url)
    response = client.completions.create(
    model="Qwen/Qwen3-Reranker-0.6B",
    prompt=messages,
    top_p=0.95,
    max_tokens=1,
    logprobs=20,
    extra_body={"allowed_token_ids": [true_token_id, false_token_id]})
    print(response.choices[0].logprobs.token_logprobs[-1])

    final_logits = response.choices[0].logprobs.top_logprobs[-1]
    print("final_logits", final_logits)
    tokens = response.choices[0].logprobs.tokens

    if true_token not in final_logits:
        true_logit = -10
    else:
        true_logit = final_logits[true_token]
    if false_token not in final_logits:
        false_logit = -10
    else:
        false_logit = final_logits[false_token]
    
    print(true_token, true_logit)
    print(false_token, false_logit)



if __name__ == "__main__":
    main()