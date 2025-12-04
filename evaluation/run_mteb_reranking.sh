export OPENBLAS_NUM_THREADS='8'
export VLLM_USE_MODELSCOPE=False

model_path="Qwen/Qwen3-Reranker-8B"
model_name="Qwen/Qwen3-Reranker-8B"
benchmark="MTEB(eng, v2)"
previous_save_path="/workspace/Qwen3-Embedding/evaluation/retrieval_data"

python run_mteb_reranking.py \
  --model ${model_path} \
  --batch_size 16 --precision fp16 \
  --backend "openai" \
  --model_kwargs "{\"batch_size\": 8}" \
  --run_kwargs "{\"save_predictions\": \"true\"}" \
  --previous_results ${previous_save_path} \
  --output_dir  results/${model_name}  \
  --benchmark "${benchmark}" \
  --tasks "Touche2020Retrieval.v3" $@ \
  --langs "eng"


