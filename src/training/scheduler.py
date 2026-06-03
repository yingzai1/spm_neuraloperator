import jax.numpy as jnp
import jax

def cosine_schedule_with_warmup(
    warmup_steps: int,
    peak_lr: float,
    total_steps: int,
    end_lr: float = 0.0
):
    """Returns an Optax schedule: warmup -> cosine decay."""
    
    def schedule_fn(step):
        step_f = jnp.asarray(step, dtype=jnp.float32)
        warmup_f = jnp.asarray(float(warmup_steps), dtype=jnp.float32)
        total_f = jnp.asarray(float(total_steps), dtype=jnp.float32)
        peak_lr_f = jnp.asarray(float(peak_lr), dtype=jnp.float32)
        end_lr_f = jnp.asarray(float(end_lr), dtype=jnp.float32)

        # Warmup phase: linear ramp from 0 → peak_lr
        lr_warm = peak_lr_f * (step_f / jnp.maximum(1.0, warmup_f))

        # Cosine decay phase
        progress = (step_f - warmup_f) / jnp.maximum(1.0, (total_f - warmup_f))
        progress = jnp.clip(progress, 0.0, 1.0)
        lr_decay = end_lr_f + (peak_lr_f - end_lr_f) * 0.5 * (1.0 + jnp.cos(jnp.pi * progress))

        lr = jnp.where(step_f < warmup_f, lr_warm, lr_decay)
        return lr.astype(jnp.float32)
    
    return schedule_fn 