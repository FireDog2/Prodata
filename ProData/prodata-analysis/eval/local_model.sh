CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
  --model your-model-path/ProData-Qwen2.5-14B \
  --served-model-name ProData-Qwen2.5-14B \
  --tensor-parallel-size 2 \
  --port 8000
