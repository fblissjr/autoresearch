"""
Performance profiling for autoresearch-mlx training.
Runs ~20 training steps (uncompiled + compiled), instruments each phase, reports breakdown.
Usage: uv run bench.py
"""

import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

from prepare import MAX_SEQ_LEN, Tokenizer, make_dataloader, get_token_bytes, EVAL_TOKENS
from train import (
    GPT, build_model_config, _sliding_window_mask_cache, _norm_weight_cache,
    loss_fn, DEPTH, DEVICE_BATCH_SIZE, TOTAL_BATCH_SIZE,
    MATRIX_LR, ADAM_BETAS, X0_BETAS, EMBEDDING_LR, UNEMBEDDING_LR,
    SCALAR_LR, WEIGHT_DECAY,
)

NUM_STEPS = 20
WARMUP_STEPS = 5

# ---------------------------------------------------------------------------
# Model init
# ---------------------------------------------------------------------------

t0 = time.perf_counter()
mx.random.seed(42)

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
config = build_model_config(DEPTH, vocab_size)

model = GPT(config)
model.init_weights()
mx.eval(model.parameters())
t_init = time.perf_counter() - t0

print(f"=== Model Init ===")
print(f"Build + init_weights + materialize: {t_init * 1000:.0f}ms")
print()

# ---------------------------------------------------------------------------
# Optimizer + dataloader
# ---------------------------------------------------------------------------

dmodel_scale = (config.n_embd / 768) ** -0.5

def is_muon_param(path, weight):
    return 'layers' in path and weight.ndim >= 2 and 've_gate' not in path
def is_embedding(path, weight):
    return 'wte' in path or 'value_embeds' in path
def is_x0_lambdas(path, weight):
    return 'x0_lambdas' in path
def is_resid_lambdas(path, weight):
    return 'resid_lambdas' in path

muon_opt = optim.Muon(learning_rate=MATRIX_LR, momentum=0.95, weight_decay=WEIGHT_DECAY)
embed_opt = optim.AdamW(learning_rate=EMBEDDING_LR * dmodel_scale, betas=list(ADAM_BETAS), eps=1e-10, weight_decay=0.0)
x0_opt = optim.AdamW(learning_rate=SCALAR_LR * dmodel_scale, betas=list(X0_BETAS), eps=1e-10, weight_decay=0.0)
resid_opt = optim.AdamW(learning_rate=SCALAR_LR * 0.01 * dmodel_scale, betas=list(ADAM_BETAS), eps=1e-10, weight_decay=0.0)
fallback_opt = optim.AdamW(learning_rate=UNEMBEDDING_LR * dmodel_scale, betas=list(ADAM_BETAS), eps=1e-10, weight_decay=0.0)
optimizer = optim.MultiOptimizer(
    [muon_opt, embed_opt, x0_opt, resid_opt, fallback_opt],
    [is_muon_param, is_embedding, is_x0_lambdas, is_resid_lambdas],
)

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")

# ---------------------------------------------------------------------------
# Training steps
# ---------------------------------------------------------------------------

print(f"=== Training ({NUM_STEPS} steps, grad_accum={grad_accum_steps}) ===")

step_data = []

for step in range(NUM_STEPS):
    t_step_start = time.perf_counter()

    # -- Data loading (all micro-steps) --
    t_data_start = time.perf_counter()
    batches = []
    for _ in range(grad_accum_steps):
        x, y, epoch = next(train_loader)
        batches.append((x, y))
    t_data = time.perf_counter() - t_data_start

    # -- Forward + backward (all micro-steps) --
    t_fwd_start = time.perf_counter()
    accumulated_grads = None
    train_loss_val = mx.array(0.0)

    for x, y in batches:
        loss, grads = loss_and_grad_fn(model, x, y)
        mx.eval(loss, grads)
        train_loss_val = train_loss_val + loss

        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            accumulated_grads = tree_map(lambda a, g: a + g, accumulated_grads, grads)

    accumulated_grads = tree_map(lambda g: g * (1.0 / grad_accum_steps), accumulated_grads)
    train_loss_val = train_loss_val * (1.0 / grad_accum_steps)
    t_fwd = time.perf_counter() - t_fwd_start

    # -- Optimizer update --
    t_opt_start = time.perf_counter()
    optimizer.update(model, accumulated_grads)
    del accumulated_grads
    mx.eval(train_loss_val, model.parameters(), optimizer.state)
    t_opt = time.perf_counter() - t_opt_start

    t_total = time.perf_counter() - t_step_start
    loss_val = train_loss_val.item()
    tok_per_sec = int(TOTAL_BATCH_SIZE / t_total)

    step_data.append({
        'data': t_data, 'fwd_bwd': t_fwd, 'optim': t_opt,
        'total': t_total, 'loss': loss_val, 'tok_per_sec': tok_per_sec,
    })

    print(f"Step {step:3d}: data={t_data*1000:6.0f}ms  fwd+bwd={t_fwd*1000:6.0f}ms  optim={t_opt*1000:5.0f}ms  total={t_total*1000:6.0f}ms  tok/s={tok_per_sec:6,}  loss={loss_val:.4f}")

# ---------------------------------------------------------------------------
# Averages (excluding warmup)
# ---------------------------------------------------------------------------

print()
print(f"=== Averages (steps {WARMUP_STEPS}-{NUM_STEPS-1}, excluding warmup) ===")
steady = step_data[WARMUP_STEPS:]
avg = {k: sum(s[k] for s in steady) / len(steady) for k in ['data', 'fwd_bwd', 'optim', 'total']}
avg['overhead'] = avg['total'] - avg['data'] - avg['fwd_bwd'] - avg['optim']
avg_tok = int(sum(s['tok_per_sec'] for s in steady) / len(steady))

for phase in ['data', 'fwd_bwd', 'optim', 'overhead']:
    pct = 100 * avg[phase] / avg['total']
    print(f"{phase:12s}: {avg[phase]*1000:7.1f}ms  ({pct:4.1f}%)")
print(f"{'total':12s}: {avg['total']*1000:7.1f}ms")
print(f"{'tok/sec':12s}: {avg_tok:,}")

# ---------------------------------------------------------------------------
# Compiled training (10 steps, compare to uncompiled)
# ---------------------------------------------------------------------------

print()
print(f"=== Compiled Training (10 steps) ===")

# Fresh 5-group optimizer for compiled path
compiled_muon = optim.Muon(learning_rate=MATRIX_LR, momentum=0.95, weight_decay=WEIGHT_DECAY)
compiled_embed = optim.AdamW(learning_rate=EMBEDDING_LR * dmodel_scale, betas=list(ADAM_BETAS), eps=1e-10, weight_decay=0.0)
compiled_x0 = optim.AdamW(learning_rate=SCALAR_LR * dmodel_scale, betas=list(X0_BETAS), eps=1e-10, weight_decay=0.0)
compiled_resid = optim.AdamW(learning_rate=SCALAR_LR * 0.01 * dmodel_scale, betas=list(ADAM_BETAS), eps=1e-10, weight_decay=0.0)
compiled_fallback = optim.AdamW(learning_rate=UNEMBEDDING_LR * dmodel_scale, betas=list(ADAM_BETAS), eps=1e-10, weight_decay=0.0)
compiled_optimizer = optim.MultiOptimizer(
    [compiled_muon, compiled_embed, compiled_x0, compiled_resid, compiled_fallback],
    [is_muon_param, is_embedding, is_x0_lambdas, is_resid_lambdas],
)

state = [model.state, compiled_optimizer.state]

@partial(mx.compile, inputs=state, outputs=state)
def compiled_step(x, y):
    loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
    compiled_optimizer.update(model, grads)
    return loss

compiled_data = []
for step in range(10):
    x, y, epoch = next(train_loader)
    t0 = time.perf_counter()
    loss = compiled_step(x, y)
    mx.eval(loss)
    dt = time.perf_counter() - t0
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    compiled_data.append({'total': dt, 'loss': loss.item(), 'tok_per_sec': tok_per_sec})
    print(f"Step {step:3d}: total={dt*1000:6.0f}ms  tok/s={tok_per_sec:6,}  loss={loss.item():.4f}")

compiled_steady = compiled_data[2:]  # skip first 2 for compilation warmup
avg_compiled_total = sum(s['total'] for s in compiled_steady) / len(compiled_steady)
avg_compiled_tok = int(sum(s['tok_per_sec'] for s in compiled_steady) / len(compiled_steady))

print(f"\nCompiled avg (steps 2-9): {avg_compiled_total*1000:.1f}ms, {avg_compiled_tok:,} tok/sec")
speedup = avg['total'] / avg_compiled_total
print(f"Speedup vs uncompiled: {speedup:.2f}x")

# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

print()
print(f"=== Memory ===")
peak_mem = mx.get_peak_memory() / 1024 / 1024
print(f"peak_memory_mb: {peak_mem:.1f}")

mask_keys = list(_sliding_window_mask_cache.keys())
print(f"mask_cache_entries: {len(mask_keys)} (keys: {mask_keys})")

norm_keys = list(_norm_weight_cache.keys())
print(f"norm_weight_cache_entries: {len(norm_keys)} (keys: {norm_keys})")

# ---------------------------------------------------------------------------
# Eval timing (2 steps)
# ---------------------------------------------------------------------------

print()
print(f"=== Eval (batch=32 vs batch=64, compiled vs uncompiled) ===")

token_bytes = get_token_bytes()

for eval_batch, label in [(DEVICE_BATCH_SIZE, "batch=32"), (64, "batch=64")]:
    val_loader = make_dataloader(tokenizer, eval_batch, MAX_SEQ_LEN, "val")
    eval_steps_full = EVAL_TOKENS // (eval_batch * MAX_SEQ_LEN)

    # Uncompiled
    eval_times = []
    for i in range(3):
        t0 = time.perf_counter()
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').reshape(-1)
        y_flat = y.reshape(-1)
        nbytes = token_bytes[y_flat]
        mask = (nbytes > 0).astype(mx.float32)
        _ = mx.sum(loss_flat * mask).item()
        _ = mx.sum(nbytes).item()
        dt = time.perf_counter() - t0
        eval_times.append(dt)

    avg_eval = sum(eval_times[1:]) / len(eval_times[1:])  # skip first
    projected = avg_eval * eval_steps_full
    print(f"  {label} uncompiled: {avg_eval*1000:.0f}ms/step, projected {eval_steps_full} steps = {projected:.0f}s")

    # Compiled forward
    compiled_model = mx.compile(model)
    val_loader = make_dataloader(tokenizer, eval_batch, MAX_SEQ_LEN, "val")
    eval_times_compiled = []
    for i in range(3):
        t0 = time.perf_counter()
        x, y, _ = next(val_loader)
        loss_flat = compiled_model(x, y, reduction='none').reshape(-1)
        y_flat = y.reshape(-1)
        nbytes = token_bytes[y_flat]
        mask = (nbytes > 0).astype(mx.float32)
        _ = mx.sum(loss_flat * mask).item()
        _ = mx.sum(nbytes).item()
        dt = time.perf_counter() - t0
        eval_times_compiled.append(dt)

    avg_compiled = sum(eval_times_compiled[1:]) / len(eval_times_compiled[1:])
    projected_compiled = avg_compiled * eval_steps_full
    print(f"  {label} compiled:   {avg_compiled*1000:.0f}ms/step, projected {eval_steps_full} steps = {projected_compiled:.0f}s")

# ---------------------------------------------------------------------------
# Loss sanity check
# ---------------------------------------------------------------------------

print()
first_loss = step_data[0]['loss']
last_loss = step_data[-1]['loss']
direction = "decreasing" if last_loss < first_loss else "NOT decreasing (check!)"
print(f"=== Sanity ===")
print(f"loss step 0: {first_loss:.4f}")
print(f"loss step {NUM_STEPS-1}: {last_loss:.4f}")
print(f"trend: {direction}")
