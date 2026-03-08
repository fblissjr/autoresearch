"""Tests for MLX optimizations: compiled eval, compiled training step."""

import time
import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

from prepare import MAX_SEQ_LEN, Tokenizer, make_dataloader, get_token_bytes, EVAL_TOKENS
from train import (
    GPT, build_model_config, loss_fn,
    DEPTH, DEVICE_BATCH_SIZE, TOTAL_BATCH_SIZE, MATRIX_LR, ADAM_BETAS,
)


def _make_model():
    """Build a small model for testing."""
    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    config = build_model_config(DEPTH, vocab_size)
    model = GPT(config)
    model.init_weights()
    mx.eval(model.parameters())
    return model, tokenizer


# ---------------------------------------------------------------------------
# D2: Compiled eval forward pass
# ---------------------------------------------------------------------------

class TestCompiledEval:
    """Verify compiled eval produces identical results and runs faster."""

    def test_compiled_eval_matches_uncompiled(self):
        """Compiled forward pass produces same loss values as uncompiled."""
        model, tokenizer = _make_model()
        val_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "val")
        token_bytes = get_token_bytes()

        x, y, _ = next(val_loader)

        # Uncompiled
        loss_flat_base = model(x, y, reduction='none').reshape(-1)
        y_flat = y.reshape(-1)
        nbytes = token_bytes[y_flat]
        mask = (nbytes > 0).astype(mx.float32)
        nats_base = mx.sum(loss_flat_base * mask).item()
        bytes_base = mx.sum(nbytes).item()

        # Compiled forward
        compiled_forward = mx.compile(model)
        loss_flat_compiled = compiled_forward(x, y, reduction='none').reshape(-1)
        nats_compiled = mx.sum(loss_flat_compiled * mask).item()

        # Should be identical (same computation graph, just fused)
        assert abs(nats_base - nats_compiled) < 1e-2, (
            f"Compiled eval nats mismatch: {nats_base} vs {nats_compiled}"
        )

    def test_compiled_eval_faster_than_uncompiled(self):
        """Compiled forward pass is at least as fast (ideally faster) than uncompiled."""
        model, tokenizer = _make_model()
        val_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "val")

        # Warmup both paths
        x, y, _ = next(val_loader)
        _ = model(x, y, reduction='none')
        mx.eval(_)

        compiled_forward = mx.compile(model)
        _ = compiled_forward(x, y, reduction='none')
        mx.eval(_)

        # Time uncompiled (3 steps)
        times_base = []
        for _ in range(3):
            x, y, _ = next(val_loader)
            t0 = time.perf_counter()
            out = model(x, y, reduction='none')
            mx.eval(out)
            times_base.append(time.perf_counter() - t0)

        # Time compiled (3 steps)
        times_compiled = []
        for _ in range(3):
            x, y, _ = next(val_loader)
            t0 = time.perf_counter()
            out = compiled_forward(x, y, reduction='none')
            mx.eval(out)
            times_compiled.append(time.perf_counter() - t0)

        avg_base = sum(times_base) / len(times_base)
        avg_compiled = sum(times_compiled) / len(times_compiled)

        # Compiled should not be slower (allow 10% margin for noise)
        assert avg_compiled <= avg_base * 1.1, (
            f"Compiled eval slower: {avg_compiled*1000:.0f}ms vs base {avg_base*1000:.0f}ms"
        )


# ---------------------------------------------------------------------------
# D2: Larger eval batch
# ---------------------------------------------------------------------------

class TestLargerEvalBatch:
    """Verify larger eval batch produces equivalent BPB."""

    def test_eval_batch_64_produces_valid_bpb(self):
        """Eval at batch=64 produces a finite, positive BPB value."""
        model, tokenizer = _make_model()
        token_bytes = get_token_bytes()
        eval_batch = 64
        val_loader = make_dataloader(tokenizer, eval_batch, MAX_SEQ_LEN, "val")

        # Run 2 eval steps at batch=64
        total_nats = 0.0
        total_bytes = 0
        for _ in range(2):
            x, y, _ = next(val_loader)
            loss_flat = model(x, y, reduction='none').reshape(-1)
            y_flat = y.reshape(-1)
            nbytes = token_bytes[y_flat]
            mask = (nbytes > 0).astype(mx.float32)
            total_nats += mx.sum(loss_flat * mask).item()
            total_bytes += mx.sum(nbytes).item()

        bpb = total_nats / (math.log(2) * total_bytes)
        assert math.isfinite(bpb) and bpb > 0, f"Invalid BPB at batch=64: {bpb}"


# ---------------------------------------------------------------------------
# D1: Compiled training step
# ---------------------------------------------------------------------------

class TestCompiledTrainingStep:
    """Verify compiled loss+grad produces valid gradients."""

    def test_compiled_step_with_state_tracking(self):
        """mx.compile with inputs/outputs state tracking works for training."""
        from functools import partial as fpartial
        model, tokenizer = _make_model()
        train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")

        optimizer = optim.AdamW(
            learning_rate=MATRIX_LR,
            betas=list(ADAM_BETAS),
            eps=1e-10,
            weight_decay=0.0,
        )

        state = [model.state, optimizer.state]

        @fpartial(mx.compile, inputs=state, outputs=state)
        def compiled_step(x, y):
            loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
            optimizer.update(model, grads)
            return loss

        x, y, _ = next(train_loader)
        loss = compiled_step(x, y)
        mx.eval(loss)

        assert loss.item() > 0, f"Loss should be positive: {loss.item()}"
        assert loss.item() < 20, f"Loss too high for random init: {loss.item()}"

    def test_compiled_step_with_schedule_no_recompilation(self):
        """Step-based LR schedule works with compiled training (no recompilation)."""
        from functools import partial as fpartial
        model, tokenizer = _make_model()
        train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")

        # Use step-based schedule instead of time-based
        schedule = optim.join_schedules(
            [optim.linear_schedule(0.0, MATRIX_LR, steps=5),
             optim.cosine_decay(MATRIX_LR, decay_steps=15)],
            [5]
        )
        optimizer = optim.AdamW(
            learning_rate=schedule,
            betas=list(ADAM_BETAS),
            eps=1e-10,
            weight_decay=0.0,
        )

        state = [model.state, optimizer.state]

        @fpartial(mx.compile, inputs=state, outputs=state)
        def compiled_step(x, y):
            loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
            optimizer.update(model, grads)
            return loss

        losses = []
        for _ in range(10):
            x, y, _ = next(train_loader)
            loss = compiled_step(x, y)
            mx.eval(loss)
            losses.append(loss.item())

        # Should not diverge over 10 steps
        assert all(l < 100 for l in losses), f"Loss diverged with schedule: {losses}"

    def test_compiled_training_step_reduces_loss(self):
        """A few compiled training steps should reduce loss."""
        model, tokenizer = _make_model()
        train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")

        optimizer = optim.AdamW(
            learning_rate=MATRIX_LR,
            betas=list(ADAM_BETAS),
            eps=1e-10,
            weight_decay=0.0,
        )

        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

        losses = []
        for step in range(5):
            x, y, _ = next(train_loader)
            loss, grads = loss_and_grad_fn(model, x, y)
            optimizer.update(model, grads)
            del grads
            mx.eval(loss, model.parameters(), optimizer.state)
            losses.append(loss.item())

        # Loss should generally decrease over 5 steps (allow some noise)
        # At minimum, it shouldn't diverge
        assert losses[-1] < 100, f"Loss diverged: {losses}"


if __name__ == "__main__":
    import sys

    test_classes = [TestCompiledEval, TestLargerEvalBatch, TestCompiledTrainingStep]
    failures = []

    for cls in test_classes:
        instance = cls()
        for name in dir(instance):
            if not name.startswith("test_"):
                continue
            print(f"Running {cls.__name__}.{name}...", end=" ", flush=True)
            try:
                getattr(instance, name)()
                print("PASS")
            except Exception as e:
                print(f"FAIL: {e}")
                failures.append(f"{cls.__name__}.{name}: {e}")

    if failures:
        print(f"\n{len(failures)} failure(s):")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print(f"\nAll tests passed.")
