from transformers import AutoTokenizer

instruction = "Given a web search query, retrieve relevant passages that answer the query"
model = "Qwen/Qwen3-Reranker-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model)

prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n'
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
chat_template_suffix ="<|im_start|>assistant\n<think>\n\n</think>\n\n" 
chat_template_suffix_token = tokenizer.encode(chat_template_suffix, add_special_tokens=False)
query_template = "{prefix}<Instruct>: {instruction}\n\n<Query>: {query}\n\n"
document_template = "<Document>: {doc}{suffix}"


queries = [
    "What is the capital of France?"
]

documents = [
    "The capital of Brazil is Brasilia.",
    "The capital of France is Paris.",
    "What is the capital of France?",
    "Horses and cows are both animals.",
]


def format_instruction(instruction, query, doc):
    if isinstance(query, tuple):
        instruction = query[0]
        query = query[1]
    text = [
        {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
        {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
    ]

    return text

pairs = list(zip(queries*len(documents), documents))

max_length= 80
input_queries = [query_template.format(prefix=prefix, instruction=instruction, query=query) for query, doc in pairs] #67        
#print(input_queries[0])
input_queries_len = [tokenizer.encode(query, add_special_tokens=False) for query in input_queries]
#print(input_queries_len[0])


## doc
docs = [document_template.format(doc=doc, suffix=suffix) for query, doc in pairs]

docs_len = [tokenizer.encode(doc, add_special_tokens=False) for doc in docs]
print(input_queries_len[0]+ docs_len[0])
print("=====================================")
##
# truncate documents if needed 
full_sentences = [format_instruction(instruction, query, doc) for query, doc in pairs]

full_sentences =  tokenizer.apply_chat_template(
    full_sentences, tokenize=True, add_generation_prompt=False, enable_thinking=False
)

print("chat_template_suffix_token", chat_template_suffix_token)
full_sentences = [e + chat_template_suffix_token for e in full_sentences]
print(full_sentences[0])

print(input_queries_len[0]+docs_len[0] == full_sentences[0])
#print([input_queries_len[i] + docs_len[i] for i in range(len(docs))])
#print([len(ele) for ele in full_sentences])

# full_sentences = [ele[:self.max_length] + self.suffix_tokens for ele in full_sentences]
# input_docs_truncated = [sentence[q_len:] for sentence, q_len in zip(full_sentences, input_queries_len)]
# input_docs_truncated = [self.tokenizer.decode(ele, skip_special_tokens=True) for ele in input_docs_truncated]