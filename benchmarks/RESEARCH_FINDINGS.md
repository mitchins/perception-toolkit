# Research Findings

## Scope

These findings summarize the current local image-description benchmark work for the sidecar server.

The benchmark uses the seed set in `test_resources/` and scores captions on whether they surface what matters for downstream tool-calling models:

- medium or non-photographic status
- subject
- scene or composition
- color treatment
- text
- special details that change interpretation

This is a dimension-based benchmark, not a single-reference-caption benchmark.

## Results

### Best Florence Variants

Florence task-token comparison on the 13-image seed set:

| Model | Score | Critical | Problems | Avg words |
| --- | ---: | ---: | ---: | ---: |
| `Florence-2-large + more-detailed-caption` | `0.535` | `18/26` | `0` | `77.9` |
| `Florence-2-base + more-detailed-caption` | `0.535` | `18/26` | `0` | `82.8` |
| `Florence-2-large-ft + more-detailed-caption` | `0.359` | `13/26` | `0` | `24.6` |
| `Florence-2-large + detailed-caption` | `0.343` | `13/26` | `0` | `22.9` |
| `Florence-2-base-ft + more-detailed-caption` | `0.339` | `12/26` | `0` | `28.9` |
| `Florence-2-base + detailed-caption` | `0.336` | `12/26` | `0` | `22.5` |
| `Florence-2-base + caption` | `0.240` | `10/26` | `0` | `4.3` |

Key Florence result:

- `more-detailed-caption` is the only Florence task that consistently retains enough useful detail.
- Plain pretrained `base` and `large` tie.
- The `-ft` checkpoints are materially shorter and less complete for this benchmark.

### Florence vs Moondream

Head-to-head on the same 13-image seed set:

| Model | Score | Critical | Problems | Avg words |
| --- | ---: | ---: | ---: | ---: |
| `Moondream2 official + caption-long` | `0.668` | `23/26` | `0` | `93.5` |
| `Moondream2 official + query` | `0.622` | `22/26` | `0` | `31.8` |
| `Florence-2-base + more-detailed-caption` | `0.535` | `18/26` | `0` | `82.8` |

### Lightweight / Control Backends

Earlier baselines on the same benchmark:

| Model | Score | Critical | Problems | Avg words |
| --- | ---: | ---: | ---: | ---: |
| `BLIP base` | `0.425` | `17/26` | `0` | `9.2` |
| `ClipCap conceptual` | `0.296` | `13/26` | `0` | `8.2` |
| `SmolVLM-256M` | `0.128` | `9/26` | `9` | `43.5` |

Additional notes:

- `ClipCap` behaved like a sane lightweight control, but was too generic to compete.
- `PaliGemma 3B mix 448` ran, but its tested modes were too generic to justify more benchmark time right now.
- The initial patched/offline Moondream path produced garbage and should not be treated as a model-quality result. The meaningful Moondream results are the official-loader runs.

## Reasoning

### Why Florence Did Not Win

Florence remains useful, but it plateaued:

- It tends to under-call non-photographic medium and style.
- `caption` and `detailed-caption` are too compressed for this task.
- The `-ft` checkpoints often produce fragmentary, terse outputs that look dense but omit too much subject and scene information.
- `more-detailed-caption` improves usefulness, but still misses some of the exact distinctions that motivated the work, especially texture/asset and illustration cues.

The best Florence setting is therefore:

- `microsoft/Florence-2-base`
- task: `more-detailed-caption`

`Florence-2-large` did not beat `base` on this benchmark, so scale alone did not solve the problem.

### Why Moondream Won

Moondream covered more critical distinctions and did a better job surfacing what changes interpretation:

- better medium and texture recognition on important non-photographic cases
- stronger recovery of black-and-white, text-heavy, and texture/asset cases
- better overall critical coverage than Florence

However, the two Moondream modes are not equivalent qualitatively.

#### `caption-long`

Strengths:

- highest benchmark score
- richer scene detail
- more likely to include composition and contextual cues

Weaknesses:

- often padded with decorative prose
- more willing to invent unsupported specifics
- clipped mid-sentence on `12/13` benchmark images in the current runs

#### `query`

Strengths:

- much shorter and cleaner
- usually complete, not truncated
- more production-friendly for tool-calling LLMs
- still clearly beats Florence on benchmark quality

Weaknesses:

- can omit some scene or composition nuance that `caption-long` retains
- still misses medium on some hard cases
- one severe hallucination matters more than the raw score suggests: it described the llama image as a lioness attacking a llama

### Holistic Conclusion

The benchmark score alone is not the full reason for the decision.

`caption-long` wins numerically, but it is not the best operational output format because:

- it is overly verbose
- it is frequently truncated
- it hallucinates more decorative specifics

`query` is the better default because it keeps most of the quality gain while producing a cleaner, more controllable artifact for downstream LLM use.

## Decision

### Current Decision

Primary backend choice for the sidecar:

- `Moondream2`
- official loader
- `query` mode

Fallback backend:

- `Florence-2-base`
- `more-detailed-caption`

Optional quality/debug mode:

- `Moondream2`
- official loader
- `caption-long`

### Why This Decision

- `Moondream2 query` beats Florence clearly on the benchmark.
- It is much shorter and more usable than `caption-long`.
- Florence remains valuable as a stable fallback and comparison baseline.

### Next Decision Already Emerging

The next architectural lever is a cheap medium/type prepass before captioning.

That prepass should:

- predict a soft `medium_hint` such as photograph, illustration, render, texture/background asset, or graphic/text-heavy
- feed that hint into Moondream or the reasoning LLM
- remain advisory, not a hard router at first

The current likely path is:

1. add structured `medium_hint` output
2. use it to steer Moondream query prompts
3. only later consider routing to medium-specific caption backends or fine-tunes
