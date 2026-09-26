# Neural Painter Research Agent Rules

The autonomous agent exists only to advance the Neural Painter research program.

## Scientific discipline

1. Implement one falsifiable mutation at a time.
2. Do not simultaneously change representation, loss, renderer, and evaluation.
3. Prefer the smallest experiment that distinguishes the current hypothesis.
4. Preserve deterministic seeds and existing experiment methods.
5. Never weaken or delete tests merely to make a mutation pass.
6. Never optimize solely for MSE; reconstruction, structure, clutter, and runtime all matter.
7. Preserve compatibility with arbitrary future video inputs. Do not introduce symmetry or object-template priors.
8. Treat renderer, metrics, diagnostics, research-controller, and autonomous-harness code as protected infrastructure.
9. Rejected hypotheses count as evidence. Do not retry an equivalent mutation unless new evidence justifies it.
10. Do not modify generated outputs, benchmark histories, or previous experiment records.
11. Prefer changes to painter mechanisms and focused tests over infrastructure expansion.
12. Finish once the requested mutation is implemented and locally validated; the outer harness decides scientific acceptance.

## Allowed editing scope

The agent may edit:
- painter modules that implement painter mechanisms, except protected infrastructure,
- scripts needed to expose a new experimental method,
- focused tests,
- README.md or pyproject.toml only when required by the mutation.

The agent has no arbitrary shell tool. It receives only constrained read/search/patch/test/diff operations.
