# Benchmark Next Steps

## Current State

The current benchmark picture is stable enough to make short-term planning decisions:

- `Moondream2 official + query` is the best current default.
- `Florence-2-base + more-detailed-caption` is the best Florence fallback.
- `Moondream 0.5B int8 + caption-normal` is interesting, but it did not beat the 2B Moondream path.
- None of the tested captioners reliably and explicitly call out `texture/background asset` strongly enough on the hardest non-photographic examples.

## Current Decision

Use this as the working decision until new benchmark data changes it:

- primary sidecar backend: `Moondream2 official + query`
- fallback backend: `Florence-2-base + more-detailed-caption`
- optional debug mode: `Moondream2 official + caption-long`

Do not promote `Moondream 0.5B` to the default path yet.

## Why

- The 2B Moondream family already won the benchmark on quality.
- The 0.5B legacy local runtime was harder to stand up and still landed below the 2B result.
- The 0.5B `query` path was clearly weaker than its own `caption-normal` path, which is a bad sign for downstream tool-description use.
- The remaining quality gap is not obviously solved by shrinking the captioner. It looks more like a missing `medium/type` prior problem.

## Immediate Plan

1. Keep the current Apple Silicon benchmark result as the reference quality baseline.
2. Benchmark `moondream/moondream-2b-2025-04-14-4bit` on production CUDA hardware.
3. Compare it directly against the current 2B query baseline on both quality and warmed latency.
4. If the quality delta is negligible, treat the 4-bit 2B model as the primary deployment target.
5. Add a cheap `medium_hint` prepass and compare raw Moondream query vs hinted Moondream query.

## Deferred Plan

These are lower priority until the 2B 4-bit CUDA result is known:

- benchmark `Moondream 0.5B int4`
- revisit Florence ONNX as a deployment optimization experiment
- evaluate medium-specific routing only after a structured `medium_hint` exists

## Guardrails

- Do not assume a smaller model is automatically the better production choice if it gives away the exact distinctions that matter to tool use.
- Do not keep expanding the benchmark matrix without a clear decision question.
- Do not absorb image generation concerns into this repo; keep the focus on perception and structured image understanding.
