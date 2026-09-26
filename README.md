# Neural Painter

Research prototype for **reference-conditioned, persistent stroke-space neural painting**.

## Phase 0

Phase 0 tests whether an explicit stroke representation is a viable state space before introducing learned models.

Several Phase 0 methods are available, including a differentiable refinement stage:

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

For differentiable refinement:

```bash
python -m pip install -e ".[dev,refine]"
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
4. Differentiable local stroke refinement — implemented
5. Reference-conditioned stroke policies
6. Persistent stroke transport across video frames
7. Learned real-time neural painter

The central research hypothesis is that explicit persistent strokes can support reference-controlled mark-making and stronger temporal coherence than independent frame-wise repainting.


## Differentiable refinement

The `refined` method starts from the residual painter, selects strokes centered on the highest-error regions, and optimizes only width, color, and opacity with PyTorch through a low-resolution differentiable soft rasterizer. Stroke geometry, count, and ordering are preserved.

```bash
python scripts/run_budget_experiment.py assets/inputs/example.jpg \
  --output-dir outputs/phase0_refined \
  --budgets 100 250 500 1000 2000 \
  --palette-size 8 \
  --seed 0 \
  --method refined
```

The refinement metadata stored in `metrics.json` includes the number of optimized strokes, optimization steps, and the initial/final differentiable loss.


### GPU refinement

Refinement defaults to `--device auto`. On a local machine with CUDA-enabled PyTorch and an NVIDIA GPU, it uses CUDA automatically; CI continues to use CPU for portability.

To force CUDA:

```bash
python scripts/run_budget_experiment.py assets/inputs/fleur_de_lis.png \
  --output-dir outputs/round3/fleur_de_lis_refined \
  --budgets 500 \
  --palette-size 8 \
  --seed 0 \
  --method refined \
  --device cuda
```

Use `--device cpu` to force CPU execution.


### Structure-aware refinement

The `refined_structure` method keeps the same conservative appearance-only parameterization but replaces pure MSE with a composite objective:

```text
MSE + 0.20 * SSIM loss + 0.10 * Sobel edge loss
```

This experiment tests whether optimizing local structural similarity and edge fidelity can improve the final raster reconstruction where pure MSE refinement previously produced only marginal PSNR/MSE gains and slightly worse SSIM.

```bash
python scripts/run_budget_experiment.py assets/inputs/fleur_de_lis.png \
  --output-dir outputs/round4/fleur_de_lis_refined_structure_500 \
  --budgets 500 \
  --palette-size 8 \
  --seed 0 \
  --method refined_structure \
  --device cuda
```


### Global reconstruction-capacity audit

The `global_refined` and `global_refined_structure` methods test whether the
current stroke representation is limited mainly by optimization/allocation or by
the primitive itself.

They start from the residual painter, select many high-error strokes (or all
strokes), and optionally optimize bounded control-point motion:

```text
p = clamp(p_init + geometry_bound * tanh(delta), 0, 1)
```

This permits local geometric correction without the long-range drift observed in
the first unconstrained refinement experiment. Width and opacity are optimized
jointly, and `--continuous-color` allows optimized strokes to move off the
initial palette.

Use `--optimized-stroke-count 0` to optimize all strokes.

```bash
python scripts/run_budget_experiment.py assets/inputs/fleur_de_lis.png \
  --output-dir outputs/round5/fleur_global_refined_500 \
  --budgets 500 \
  --palette-size 8 \
  --seed 0 \
  --method global_refined \
  --device cuda \
  --optimized-stroke-count 256 \
  --optimize-geometry \
  --geometry-bound 0.03 \
  --continuous-color
```

Use `global_refined_structure` for the MSE + SSIM + Sobel-edge objective.


### Staged global refinement

The `staged_refined` and `staged_refined_structure` methods progressively
repair a large stroke program in disjoint batches. After each stage, the current
program is rerendered, residual error is recomputed, and the next highest-error
unseen strokes are selected.

This keeps GPU memory bounded while allowing much more than 256 strokes to be
optimized across a 2000-stroke program.

```bash
python scripts/run_budget_experiment.py assets/inputs/fleur_de_lis.png \
  --output-dir outputs/round6/fleur_staged_refined_2000 \
  --budgets 2000 \
  --palette-size 8 \
  --seed 0 \
  --method staged_refined \
  --device cuda \
  --optimized-stroke-count 256 \
  --refinement-stages 4 \
  --optimize-geometry \
  --geometry-bound 0.03 \
  --continuous-color
```

With the defaults above, four stages optimize up to 1024 distinct strokes.


### Repeated full-program sweeps

Staged refinement can revisit the full stroke program multiple times with
`--refinement-sweeps`. Within each sweep, batches are disjoint; at the start
of the next sweep, all strokes become eligible again and are reranked from the
new residual image. This lets early batches be corrected after later strokes
have changed.

For a 2000-stroke program with 256-stroke batches, eight stages cover the whole
program once. Two sweeps therefore revisit the full program twice:

```bash
python scripts/run_budget_experiment.py assets/inputs/fleur_de_lis.png \
  --output-dir outputs/round7/fleur_staged_structure_2000_sweep2 \
  --budgets 2000 \
  --palette-size 8 \
  --seed 0 \
  --method staged_refined_structure \
  --device cuda \
  --optimized-stroke-count 256 \
  --refinement-stages 8 \
  --refinement-sweeps 2 \
  --optimize-geometry \
  --geometry-bound 0.03 \
  --continuous-color
```


### Richer stroke-space representation

Phase 0F introduces two primitives beyond the original constant-width Bezier:

- `TaperedStroke`: quadratic Bezier centerline with independent start, middle,
  and end widths.
- `EllipsePatch`: filled oriented ellipse for broad smooth image regions.

The `rich_residual` experiment uses broad patches first on smooth high-residual
areas, then adds tapered gradient-aligned strokes for the remaining detail.

```bash
python scripts/run_budget_experiment.py assets/inputs/fleur_de_lis.png \
  --output-dir outputs/round8/fleur_rich_residual_2000 \
  --budgets 2000 \
  --palette-size 8 \
  --seed 0 \
  --method rich_residual
```

This is the first direct representation-capacity comparison against the old
constant-width stroke program at the same primitive budget.


### Region-fitted patches

The `region_rich_residual` method replaces randomly sampled area patches with
connected residual-region fitting:

1. compute residual error and smoothness,
2. threshold high-scoring smooth residual regions,
3. extract connected components,
4. fit an oriented ellipse from component covariance,
5. accept the patch only when its alpha-composited color reduces local RGB error,
6. return unused patch budget to tapered detail strokes.

```bash
python scripts/run_budget_experiment.py assets/inputs/fleur_de_lis.png \
  --output-dir outputs/round9/fleur_region_rich_2000 \
  --budgets 2000 \
  --palette-size 8 \
  --seed 0 \
  --method region_rich_residual
```

This directly tests whether the poor first rich-painter result came from the
primitive vocabulary or from uninformed patch placement.


### Rich differentiable refinement

The `region_rich_refined` and `region_rich_refined_structure` methods start
from the region-fitted rich representation and optimize both primitive families.

Ellipse patches optimize bounded center motion, radii, angle, continuous RGB,
and opacity. Tapered strokes optimize bounded Bezier control-point motion,
start/middle/end widths, continuous RGB, and opacity.

```bash
python scripts/run_budget_experiment.py assets/inputs/fleur_de_lis.png \
  --output-dir outputs/round10/fleur_region_rich_refined_2000 \
  --budgets 2000 \
  --palette-size 8 \
  --seed 0 \
  --method region_rich_refined_structure \
  --device cuda \
  --optimized-stroke-count 256 \
  --geometry-bound 0.03
```

This tests whether the richer primitive vocabulary can surpass the old
constant-width representation once both receive differentiable optimization.


### Renderer consistency calibration

The rich differentiable renderer is calibrated directly against the deterministic
Pillow rasterizer. Rich soft rendering now uses ordered alpha compositing,
pixel-scale hard-edge approximations, 24+ curve samples, and ellipse edge
softness expressed in pixel units rather than relative radius units.

Run:

```bash
python scripts/calibrate_renderer.py
```

The command reports MSE, PSNR, and SSIM between identical rich primitive
programs rendered by the real and differentiable renderers. CI also enforces a
minimum consistency threshold so future refinement changes cannot silently
increase surrogate-renderer mismatch.


### Ordered mixed-program refinement

Rich refinement now evaluates optimized primitives inside the actual compositing
program instead of against an incomplete canvas.

During patch optimization, the later tapered-stroke suffix remains present via a
cached raster affine transform. During tapered-stroke optimization, fixed strokes
before, between, and after selected trainable strokes are cached as raster
segments and reapplied in the original order.

Renderer calibration now reports both a small reference program and a denser
overlap stress test:

```bash
python scripts/calibrate_renderer.py
```

This is intended to prevent surrogate-loss improvements that do not transfer to
the final Pillow raster output.


### Structure-first rich cleanup schedule

The `region_rich_refined_schedule` method performs two ordered-context passes:

1. structure-aware refinement over a large high-residual tapered-stroke subset,
2. rerender + residual recomputation,
3. MSE cleanup refinement over a fresh large subset.

Both passes preserve fixed primitive order through cached raster segments.

```bash
python scripts/run_budget_experiment.py assets/inputs/fleur_de_lis.png \
  --output-dir outputs/round13/fleur_region_rich_schedule_2000 \
  --budgets 2000 \
  --palette-size 8 \
  --seed 0 \
  --method region_rich_refined_schedule \
  --device cuda \
  --optimized-stroke-count 512 \
  --geometry-bound 0.03
```


### Geometry-safe cleanup

The scheduled rich refiner now separates structure-stage and cleanup-stage
controls. The recommended cleanup is appearance-focused: keep control-point
geometry fixed while optimizing a small high-residual subset for MSE.

```bash
python scripts/run_budget_experiment.py assets/inputs/fleur_de_lis.png \
  --output-dir outputs/round16/fleur_region_rich_safe_cleanup_2000 \
  --budgets 2000 \
  --palette-size 8 \
  --seed 0 \
  --method region_rich_refined_schedule \
  --device cuda \
  --optimized-stroke-count 256 \
  --geometry-bound 0.02 \
  --cleanup-stroke-count 64 \
  --cleanup-geometry-bound 0.0
```

Use `--cleanup-optimize-geometry` only for explicit geometry-cleanup ablations.
When enabled, `--cleanup-geometry-drift-weight` penalizes control-point drift
from the structure-stage result.


### Polygon regions + contour-aware loss

The next representation upgrade avoids any symmetry or object-template prior.
It remains local and therefore compatible with later video transport.

- `PolygonPatch` approximates arbitrary connected residual regions with compact
  contour polygons instead of forcing every smooth mass into an ellipse.
- `polygon_rich_residual` uses accepted polygon regions first, then tapered
  Bezier strokes for detail.
- `polygon_rich_refined_contour` keeps polygon regions raster-exact and refines
  high-error tapered strokes with a stronger contour objective.

The contour objective combines:

```text
MSE
+ multi-scale SSIM
+ Sobel edge magnitude
+ gradient-orientation alignment
+ Laplacian detail matching
```

Run the first direct comparison at the same 2000-primitive budget:

```bash
python scripts/run_budget_experiment.py assets/inputs/fleur_de_lis.png \
  --output-dir outputs/round23/fleur_polygon_contour_2000 \
  --budgets 2000 \
  --palette-size 8 \
  --seed 0 \
  --method polygon_rich_refined_contour \
  --device cuda \
  --optimized-stroke-count 256 \
  --geometry-bound 0.02
```


### Bezier ribbons + positional contour loss

The mixed rich representation adds a fourth primitive family without imposing
symmetry or object templates:

- `PolygonPatch`: arbitrary broad connected regions,
- `BezierRibbon`: cubic centerline + varying width for curved coherent bands,
- `TaperedStroke`: fine local detail.

`mixed_rich_residual` fits elongated connected residual components with
Bezier ribbons first, fills remaining broad regions with polygons, then allocates
the rest of the budget to tapered detail strokes.

`mixed_rich_refined_positional` keeps the broad polygon/ribbon structure
raster-exact and refines selected detail strokes with a positional contour
objective:

```text
MSE
+ multi-scale SSIM
+ Sobel magnitude
+ edge orientation
+ Laplacian detail
+ target-edge distance penalty
+ weak excess-contour penalty
```

The positional term penalizes rendered contour energy according to its distance
from target contours, so a fuzzy halo near the correct boundary is no longer
treated as equivalent to an accurately placed edge.

Run both 2000-primitive comparisons:

```bash
python scripts/run_budget_experiment.py assets/inputs/fleur_de_lis.png \
  --output-dir outputs/round24/fleur_mixed_rich_residual_2000 \
  --budgets 2000 \
  --palette-size 8 \
  --seed 0 \
  --method mixed_rich_residual
```

```bash
python scripts/run_budget_experiment.py assets/inputs/fleur_de_lis.png \
  --output-dir outputs/round24/fleur_mixed_rich_positional_2000 \
  --budgets 2000 \
  --palette-size 8 \
  --seed 0 \
  --method mixed_rich_refined_positional \
  --device cuda \
  --optimized-stroke-count 256 \
  --geometry-bound 0.02
```
