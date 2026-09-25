# Neural Painter

Research prototype for **reference-conditioned, persistent stroke-space neural painting**.

## Phase 0

Phase 0 tests whether an explicit stroke representation is a viable state space before introducing learned models.

Initial pipeline:

```text
input image
  -> CIELAB palette extraction
  -> image-gradient estimation
  -> residual-weighted stroke placement
  -> quadratic Bezier stroke rendering
  -> painted reconstruction
```

The first experiments will measure reconstruction quality, perceptual similarity, structural preservation, stroke budget, and rendering cost.

## Planned progression

1. Procedural stroke renderer
2. Residual-driven image painting
3. Differentiable stroke optimisation
4. Reference-conditioned stroke policies
5. Persistent stroke transport across video frames
6. Learned real-time neural painter

The central research hypothesis is that explicit persistent strokes can support reference-controlled mark-making and stronger temporal coherence than independent frame-wise repainting.
