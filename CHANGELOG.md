# Changelog

## 0.2.1

- Cache sliding window attention masks (avoid recomputation every forward pass)
- Evaluate gradients per micro-step to reduce peak memory in gradient accumulation
- Include loss in mx.eval call to avoid potential double-evaluation
- Remove manual GQA head repeat (SDPA handles mismatched head counts natively)
- Use Python scalar in loss denominator (avoid mx.array type promotion)
- Release accumulated_grads before evaluation to reduce peak memory

## 0.2.0

- Port from PyTorch/CUDA to pure MLX for Apple Silicon
- Replace Flash Attention 3 with mx.fast.scaled_dot_product_attention
- Replace manual RoPE with mx.fast.rope
- Replace manual RMS norm with mx.fast.rms_norm
- Replace custom MuonAdamW with MLX AdamW (Muon via MultiOptimizer planned)
- Replace torch data loading with numpy-based packing + mx.array conversion
- Remove CUDA-specific code (pinned memory, device transfers, autocast, torch.compile)
- Switch token_bytes storage from .pt to .npy format
- Reduce default batch size for Apple Silicon memory constraints

## 0.1.0

- Initial PyTorch/CUDA implementation
- Custom GPT with Value Embeddings, RoPE, sliding window attention
- MuonAdamW optimizer (Muon + AdamW hybrid)
- 5-minute fixed time budget training
- BPB evaluation metric
