# Lessons From `imgai-server`

This note captures the parts of the older `imgai-server` project that are still relevant to `perception-toolkit`.

The goal is not to recreate that system. The goal is to keep the useful architectural lessons and avoid repeating the scope mistakes.

## What Was Worth Keeping

### 1. Service Boundary Over UI-Specific Glue

The older project was most successful when it acted as a generic HTTP service rather than a UI-specific integration.

That remains the right direction here:

- `perception-toolkit` should expose stable perception APIs
- Open WebUI can call those APIs
- ComfyUI can call those APIs
- neither client should own the core perception logic

### 2. Thin Routers, Real Service Layer

Request parsing and response shaping should stay in routers.
Model selection, loading, inference, and routing decisions should live in service modules.

This matters because the project is already starting to accumulate:

- model-specific quirks
- device/runtime logic
- prompt variants
- future classifier and routing logic

If those concerns leak into route handlers, the codebase will get messy quickly.

### 3. Medium-Aware Presets And Routing

The best reusable idea from the old project was not generation. It was the idea of switching pipelines based on image type.

In `imgai-server`, face handling already distinguished between:

- `photo`
- `anime`
- `cg`

That is directly relevant to this project.

The current version of that idea for `perception-toolkit` is:

- predict a soft `medium_hint`
- use it to improve prompting
- later, use it to steer backend choice or specialist models

### 4. Backend Strategy As A First-Class Concern

The old project treated backend selection as a real system concern instead of an implementation detail.

That is still useful here.

The project will likely need to support:

- CPU fallback
- MPS on Apple Silicon
- CUDA in deployment
- different backends with different strengths

That argues for explicit backend-selection logic instead of one hardcoded path.

### 5. Real Smoke Tests With Real Images

The old project had real-image smoke tests to catch model hallucination and bad regressions.

That instinct was correct.

The current benchmark harness is already a stronger version of that idea, and should remain part of the project rather than becoming a one-off research artifact.

## What To Avoid Repeating

### 1. Letting Generation Swallow The Project

The old project lost focus when image generation stopped being adjacent functionality and became a major product surface.

For this project:

- perception stays here
- generation stays in ComfyUI

If generation ever appears, it should be integration-level only, not a new product axis.

### 2. Building Too Much Management Plane Too Early

A model registry, cache manager, and UI for download state are useful eventually.
They are also a fast way to build a lot of code before the core pipeline is stable.

For now, `perception-toolkit` should bias toward:

- clear configs
- explicit backend wiring
- benchmark-driven decisions

Only add a heavier model-management layer once multiple stable backends genuinely need it.

### 3. Hard Routing On Weak Signals

If a cheap classifier predicts the wrong medium, and the whole system blindly routes on that result, the pipeline becomes brittle.

So medium/type prediction should start as:

- advisory
- confidence-scored
- observable

Only later should it become a stronger routing signal.

### 4. Hidden Magic

Auto-registration and broad implicit behavior are convenient at first, but they make the system harder to reason about as it grows.

For this project, explicitness is preferable while the architecture is still being shaped.

## Future Strategy For `perception-toolkit`

### Near Term

1. Keep `Moondream2 official query` as the primary caption path.
2. Keep `Florence-2-base + more-detailed-caption` as the fallback baseline.
3. Preserve the benchmark harness as the decision engine for future model changes.

### Next Likely Technical Step

Add a cheap medium/type prepass that predicts things like:

- photograph
- illustration/anime
- render
- texture/background asset
- graphic/text-heavy
- black-and-white

That output should be structured and confidence-scored.

### How To Use The Medium Prior

The medium prior can drive two different levers:

1. Prompt conditioning
   Give the captioning model or the reasoning LLM a soft hint.

2. Backend or preset selection
   Route to different prompts, presets, or models when confidence is high enough.

Those are complementary, not mutually exclusive.

### Recommended Order

1. Add structured `medium_hint`
2. Use it to steer prompts
3. Measure whether that improves benchmark outcomes
4. Only then consider backend routing or specialist fine-tunes

## Decision Guardrails

When future scope questions come up, prefer decisions that preserve these constraints:

- one clear job: perception, not generation
- stable service boundary usable by multiple clients
- structured outputs, not prose-only contracts
- benchmarked changes, not intuition-only changes
- advisory routing before hard routing
