export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS='8'
model_path="Qwen/Qwen3-Embedding-8B"
model_name="Qwen/Qwen3-Embedding-8B"
benchmark="MTEB(eng, v2)"

# overall evaluation
python run_mteb.py \
  --model ${model_path} \
  --model_name ${model_name} \
  --backend "openai" \
  --precision bf16 \
  --model_kwargs "{\"max_length\": 8192, \"attn_type\": \"causal\", \"pooler_type\": \"last\", \"do_norm\": true, \"use_instruction\": true, \"instruction_template\": \"Instruct: {}\nQuery:\", \"instruction_dict_path\": \"task_prompts.json\", \"attn_implementation\":\"flash_attention_2\"}" \
  --run_kwargs "{\"save_predictions\": \"true\"}" \
  --output_dir results/${model_name} \
  --batch_size 8 \
  --benchmark "${benchmark}" $@
