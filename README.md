# Neural Painter

Research prototype for **reference-conditioned, persistent stroke-space neural painting**.

## Phase 0

Phase 0 tests whether an explicit stroke representation is a viable state space before introducing learned models.

Two procedural methods are currently available:

```text
static baseline
input image
  -> CIELAB palette extraction
  -> image-gradient estimation
  -> gradient-biased stroke sampling
  -> quadratic Bezier stroke rendering
```

```text
residual painter
input image
  -> robust border-background estimate
  -> CIELAB palette extraction
  -> coarse-to-fine stroke passes
  -> render current canvas
  -> compute reconstruction residual
  -> residual + gradient weighted resampling
  -> repeat
```

The static method is preserved as the original baseline. The residual method adds background-aware initialization and iterative coarse-to-fine stroke allocation so the two can be compared directly.

## Setup

```bash
python -m pip install -e ".[dev]"
```

## Single reconstruction

```bash
python scripts/paint_image.py assets/inputs/example.jpg \
  --output outputs/example_500.png \
  --strokes 500 \
  --palette-size 8 \
  --seed 0
```

## Stroke-budget experiment

Static baseline:

```bash
python scripts/run_budget_experiment.py assets/inputs/example.jpg \
  --output-dir outputs/phase0_static \
  --budgets 100 250 500 1000 2000 \
  --palette-size 8 \
  --seed 0 \
  --method static
```

Residual painter:

```bash
python scripts/run_budget_experiment.py assets/inputs/example.jpg \
  --output-dir outputs/phase0_residual \
  --budgets 100 250 500 1000 2000 \
  --palette-size 8 \
  --seed 0 \
  --method residual
```

Each experiment writes the target image, one reconstruction per stroke budget, and `metrics.json` with MSE, PSNR, SSIM, elapsed rendering time, method metadata, and the canvas background used for each run.

## Planned progression

1. Procedural stroke renderer — implemented
2. Fixed-budget gradient baseline — implemented
3. Residual-driven image painting — implemented
4. Differentiable stroke optimisation
5. Reference-conditioned stroke policies
6. Persistent stroke transport across video frames
7. Learned real-time neural painter

The central research hypothesis is that explicit persistent strokes can support reference-controlled mark-making and stronger temporal coherence than independent frame-wise repainting.
