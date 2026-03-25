# Perception Sandbox — PoC

A proof-of-concept extension for **Open WebUI** that lets a **text-only, tool-calling reasoning model** (e.g. GPT-OSS) inspect uploaded images through a **local perception sidecar service**, instead of requiring native multimodal support.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Open WebUI                          │
│                                                         │
│  ┌─────────────┐   ┌─────────────────────────────────┐  │
│  │   Filter     │   │           Tools                 │  │
│  │ (inlet)      │   │  list_attachments()             │  │
│  │              │   │  inspect_image(name, intent)    │  │
│  │              │   │  extract_text(name, ...)        │  │
│  │              │   │  detect_objects(name, ...)      │  │
│  │ • intercept  │   │  tag_image(name, threshold)     │  │
│  │ • stage      │   │                                 │  │
│  │ • inject     │   │  (lightweight HTTP clients)     │  │
│  │   hint       │   │                                 │  │
│  └──────┬───────┘   └──────────┬──────────────────────┘  │
│         │                      │                         │
└─────────┼──────────────────────┼─────────────────────────┘
          │  HTTP POST           │  HTTP POST
          ▼                      ▼
┌─────────────────────────────────────────────────────────┐
│              Perception Sidecar (FastAPI)                │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Attachments  │  │  Florence-2   │  │  OCR / YOLO   │  │
│  │ Sandbox      │  │  Backend      │  │  Optional     │  │
│  │              │  │              │  │  Backends      │  │
│  │ per-session  │  │  general     │  │  text/detect   │  │
│  │ per-turn     │  │  ocr         │  │  structured    │  │
│  │ manifest     │  │  regions     │  │  signals       │  │
│  └──────────────┘  └──────────────┘  └───────────────┘  │
│                                                         │
│  GPU / CUDA inference    Config-driven backends         │
└─────────────────────────────────────────────────────────┘
```

### Separation of Concerns

| Responsibility | Where |
|---|---|
| Attachment interception | Open WebUI filter |
| Sandbox staging | Sidecar (via HTTP) |
| Tool exposure to model | Open WebUI tools |
| Model inference (Florence-2, RapidOCR, YOLO, WD-14) | Sidecar only |
| Backend configuration | Sidecar config.yaml |
| Threshold management | Sidecar |

**Hard rule**: No heavyweight inference frameworks run inside Open WebUI.

## Quick Start

### Docker Compose (recommended)

```bash
# Clone the repo
git clone <repo-url> && cd perception-sandbox

# Start everything (GPU required for default config)
docker compose up --build

# CPU-only mode:
PERCEPTION_FLORENCE_DEVICE=cpu docker compose up --build
```

Open WebUI will be available at `http://localhost:3000`.
The perception sidecar runs at `http://localhost:8200`.

### Manual Setup (development)

**1. Start the perception sidecar:**

```bash
cd perception-sandbox
pip install -r requirements-sidecar.txt
python -m perception_api.main
```

The sidecar starts on port 8200 by default.

**2. Install Open WebUI plugins:**

Copy the filter and tools into your Open WebUI instance:

- `openwebui/filter.py` → Open WebUI Functions → Add Filter
- `openwebui/tools.py` → Open WebUI Functions → Add Tool

After importing them, explicitly turn them on in Open WebUI:

- `Admin → Functions` → enable the imported filter with the small `Enabled/Disabled` switch
- `Workspace → Models → Edit` → attach the imported filter and tools to the target model

Attaching a filter to a model is not enough if the function itself is still disabled.

Set the environment variable in Open WebUI:
```
PERCEPTION_SIDECAR_URL=http://localhost:8200
```

**3. Configure a model with the tools:**

In Open WebUI, assign the imported perception tools and filter to your model (e.g. GPT-OSS).

### Benchmark Env (Local Model Comparison)

For local caption-quality benchmarking on Apple Silicon, use the dedicated Conda env:

```bash
conda env create -f environment-benchmark.yml
conda activate perception-bench-py312
```

This pins a clean Python 3.12 + `transformers` stack for Florence, Moondream, and the benchmark harness, instead of relying on whichever local env currently happens to work.

Current benchmark conclusions and backend decisions are documented in [benchmarks/RESEARCH_FINDINGS.md](benchmarks/RESEARCH_FINDINGS.md).
Longer-term architectural lessons and future strategy notes are documented in [docs/LESSONS_FROM_IMGAI_SERVER.md](docs/LESSONS_FROM_IMGAI_SERVER.md).
The current benchmark and deployment follow-up plan is documented in [docs/BENCHMARK_NEXT_STEPS.md](docs/BENCHMARK_NEXT_STEPS.md).

Example benchmark run:

```bash
python benchmarks/caption_eval.py \
  --backend moondream-local \
  --moondream-loader official \
  --moondream-mode caption-long \
  --write-predictions benchmarks/moondream2_predictions.json \
  --resume
```

Use `--offline --moondream-loader patched` only when you intentionally want to run from the local cache without any Hugging Face network fetches.

To compare Florence caption task tokens on the same checkpoint, use `--florence-task`:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python benchmarks/caption_eval.py \
  --backend florence-local \
  --device mps \
  --model-id microsoft/Florence-2-base \
  --florence-task caption \
  --write-predictions benchmarks/florence2_base_caption_predictions.json \
  --resume
```

Valid Florence task values are `caption`, `detailed-caption`, and `more-detailed-caption`.

To run the full Florence matrix in one shot, use:

```bash
benchmarks/run_florence_matrix.sh
```

The script defaults to all four Florence checkpoints and the three caption task tokens, then prints a combined leaderboard. Override behavior with env vars such as `DEVICE=mps`, `LIMIT=2`, `OFFLINE=1`, or `PYTHON_BIN=/path/to/python`, or pass explicit model ids as positional arguments.

To compare the best Florence configuration against the two sane Moondream modes, use:

```bash
benchmarks/run_florence_vs_moondream.sh
```

This runs `microsoft/Florence-2-base` with `more-detailed-caption`, plus Moondream official `caption-long` and `query`, then prints a combined leaderboard. Override with env vars such as `DEVICE=mps`, `LIMIT=2`, `OFFLINE=1`, `FLORENCE_MODEL_ID=...`, or `MOONDREAM_MODEL_ID=...`.

To compare warmed-up latency for Florence base, Florence large, and Moondream, use:

```bash
benchmarks/run_warm_latency_compare.sh
```

This performs an untimed warmup pass first, then measures steady-state per-image latency across the benchmark image set. Moondream `compile()` is enabled by default in this wrapper because that is the more realistic steady-state serving path; disable it with `MOONDREAM_COMPILE=0`. Override behavior with env vars such as `DEVICE=mps`, `LIMIT=2`, `WARMUP_PASSES=1`, `MEASURE_PASSES=2`, `MOONDREAM_MODE=query`, `MOONDREAM_MODEL_ID=...`, `OFFLINE=1`, or `WRITE_JSON=benchmarks/warm_latency_results.json`.

To fetch the official Moondream 0.5B local weights, use the Hugging Face CLI against the `vikhyatk/moondream2` repo:

```bash
hf download vikhyatk/moondream2 moondream-0_5b-int8.mf.gz \
  --revision 9dddae84d54db4ac56fe37817aeaeb502ed083e2 \
  --local-dir models/moondream
```

```bash
hf download vikhyatk/moondream2 moondream-0_5b-int4.mf.gz \
  --revision 9dddae84d54db4ac56fe37817aeaeb502ed083e2 \
  --local-dir models/moondream
```

The current official 2B benchmark path in this repo uses Hugging Face Transformers and supports `mps`. The official 0.5B local files are a different runtime story. Older official package docs described direct local inference with downloaded weights, but the current `moondream` Python package line exposes a cloud/Photon client instead of a direct local-weight loader. So downloading the 0.5B `.mf.gz` files is still useful groundwork, but it is not enough by itself to benchmark 0.5B through the current package.

The optional `moondream-package-local` backend exists as a placeholder for older or future direct-local package variants. With the current `moondream` 0.2.x client, it will raise a clear error rather than silently attempting a cloud call.

We also validated the older direct local 0.5B path in a disposable env using `moondream==0.0.5` plus the decompressed `.mf` files. That path is useful for historical comparison, but the current result does not displace the standard 2B Moondream path. The next important production benchmark is the quantized 2B CUDA path, not the legacy 0.5B ONNX path.

## Configuration

All backend configuration lives in `config.yaml` (sidecar side):

```yaml
backends:
  florence:
    enabled: true
    model_id: microsoft/Florence-2-base    # or Florence-2-large
    device: auto                            # auto, mps, cpu, cuda:0

  wd14:
    enabled: false
    model_id: SmilingWolf/wd-v1-4-convnextv2
    threshold_default: 0.35
    device: auto

  ocr:
    enabled: true
    score_threshold_default: 0.50

  detector:
    enabled: false
    model_id: yolo11n.pt
    confidence_threshold_default: 0.35
    iou_threshold_default: 0.45
    max_detections: 50
    device: auto

  classifier:
    enabled: false
    model_path: /models/custom_classifier.onnx
    threshold_default: 0.60
    device: auto

sandbox:
  base_path: /tmp/perception
  ttl_seconds: 3600
```

### Environment Variable Overrides

Every config value can be overridden:

| Variable | Default | Description |
|---|---|---|
| `PERCEPTION_FLORENCE_ENABLED` | `true` | Enable Florence-2 backend |
| `PERCEPTION_FLORENCE_MODEL_ID` | `microsoft/Florence-2-base` | HuggingFace model ID |
| `PERCEPTION_FLORENCE_DEVICE` | `auto` | Device for inference (`auto`, `mps`, `cpu`, `cuda:0`) |
| `PERCEPTION_WD14_ENABLED` | `false` | Enable WD-14 tagger |
| `PERCEPTION_WD14_THRESHOLD` | `0.35` | Default tag threshold |
| `PERCEPTION_OCR_ENABLED` | `true` | Enable RapidOCR text extraction |
| `PERCEPTION_OCR_THRESHOLD` | `0.50` | Default OCR confidence threshold |
| `PERCEPTION_DETECTOR_ENABLED` | `false` | Enable YOLO detector |
| `PERCEPTION_DETECTOR_MODEL_ID` | `yolo11n.pt` | Ultralytics checkpoint or model path |
| `PERCEPTION_DETECTOR_THRESHOLD` | `0.35` | Default detector confidence threshold |
| `PERCEPTION_DETECTOR_IOU_THRESHOLD` | `0.45` | Default NMS IoU threshold |
| `PERCEPTION_DETECTOR_MAX_DETECTIONS` | `50` | Default max detections returned |
| `PERCEPTION_DETECTOR_DEVICE` | `auto` | Device for detection (`auto`, `mps`, `cpu`, `cuda:0`) |
| `PERCEPTION_SANDBOX_BASE` | `/tmp/perception` | Sandbox base path |
| `PERCEPTION_SANDBOX_TTL` | `3600` | Sandbox TTL in seconds |

## How It Works

### End-to-end flow

1. **User uploads an image** in Open WebUI and asks "What is in this image?"
2. **Filter intercepts** the attachment, decodes it, and POSTs it to the sidecar's `/attachments/stage` endpoint.
3. **Sidecar stages** the file into a scoped sandbox directory (`/tmp/perception/<session>/<turn>/`).
4. **Filter injects** a system hint telling the model that perception tools are available.
5. **Model receives** the hint and the user's question (no raw image bytes).
6. **Model calls** `list_attachments()` → tool hits sidecar `/attachments/list` → returns manifest.
7. **Model calls** `inspect_image("image_1.jpg", intent="general")`, `extract_text("image_1.jpg")`, or `detect_objects("image_1.jpg")` depending on the task.
8. **Sidecar runs** Florence-2 for open-ended description, RapidOCR for text extraction, or YOLO for structured object counts/inventory.
9. **Model answers** the user's question using the returned evidence.

### Sandbox Security

- Files are scoped per-session and per-turn.
- Only staged files in the manifest are accessible.
- Tool inputs accept only logical names — no absolute paths.
- Path traversal is blocked at multiple levels.
- File type validation restricts to known image MIME types.
- TTL-based cleanup removes stale scopes.

## Tool Reference

### `list_attachments(scope="turn")`

Lists all attachments staged for the current turn.

```
Available attachments for this turn:
- image_1.jpg (image/jpeg, 1920x1080, 2.1 MB)
```

### `inspect_image(name, intent="general", query="")`

Runs perception on a staged image.

| Intent | Description | Backend / Task |
|---|---|---|
| `general` | Detailed image description | `<MORE_DETAILED_CAPTION>` |
| `ocr` | Text extraction | RapidOCR when enabled, otherwise Florence `<OCR>` |
| `regions` | Region-level descriptions | `<DENSE_REGION_CAPTION>` |

### `extract_text(name, threshold=0.50)`

Runs OCR on a staged image.

- Best for reading signs, labels, screenshots, menus, receipts, and UI text
- Returns extracted text lines with confidence scores
- Prefer this over generative caption models when the user explicitly wants text read

### `detect_objects(name, threshold=0.35, iou_threshold=0.45, max_detections=50)`

Runs object detection on a staged image.

- Best for counts and inventory of common object categories such as `people`, `cars`, or `pizza`
- Returns grouped counts plus raw detections with confidence scores and rough bounding boxes
- Closed-vocabulary by design, so use `inspect_image(...)` for open-ended interpretation

### `tag_image(name, threshold=0.35)`

Runs WD-14 tagging (only if enabled in config).

Returns ranked tags with confidence scores.

## Sidecar API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check with backend status |
| `/config` | GET | Read-only config summary |
| `/backends` | GET | Backend loaded/enabled status |
| `/attachments/stage` | POST | Stage a file (multipart upload) |
| `/attachments/list` | POST | List attachments for a scope |
| `/attachments/cleanup` | POST | Clean up a specific scope |
| `/attachments/cleanup-expired` | POST | Clean up all expired scopes |
| `/inspect` | POST | Run Florence-2 inspection |
| `/ocr` | POST | Run RapidOCR text extraction |
| `/detect` | POST | Run YOLO object detection |
| `/tag` | POST | Run WD-14 tagging |

## Project Structure

```
perception-sandbox/
├── openwebui/
│   ├── __init__.py
│   ├── filter.py          # Inlet filter — intercepts & stages attachments
│   └── tools.py           # Model-callable tools — HTTP clients to sidecar
├── perception_api/
│   ├── __init__.py
│   ├── main.py            # FastAPI application
│   ├── attachments.py     # Scoped sandbox management
│   ├── florence.py         # Florence-2 inference backend
│   ├── ocr.py              # RapidOCR backend (optional)
│   ├── detector.py         # YOLO detector backend (optional)
│   ├── tagger.py           # WD-14 tagger backend (optional)
│   ├── config.py           # Configuration loading
│   └── schemas.py          # Pydantic request/response models
├── config.yaml             # Backend configuration
├── docker-compose.yml      # Full stack deployment
├── Dockerfile.sidecar      # Sidecar container build
├── requirements-sidecar.txt
├── requirements-openwebui.txt
└── README.md
```

## What Is Real vs. Stubbed

### Real (fully implemented)

- ✅ Attachment interception and staging (filter)
- ✅ Scoped sandbox with manifest and path-traversal protection
- ✅ `list_attachments` tool → sidecar → manifest
- ✅ `inspect_image` tool → sidecar → Florence-2 inference
- ✅ `extract_text` tool → sidecar → RapidOCR inference
- ✅ `detect_objects` tool → sidecar → YOLO detection
- ✅ `tag_image` tool → sidecar → WD-14 inference
- ✅ Config-driven backend selection and thresholds
- ✅ Environment variable overrides
- ✅ Health/status endpoints
- ✅ Docker Compose deployment
- ✅ System hint injection for the model

### Stubbed / Minimal

- ⬜ Classifier backend (config structure present, inference not implemented)
- ⬜ Open WebUI plugin auto-registration (manual install via UI or file mount)
- ⬜ TTL cleanup runs on-demand only (no background scheduler)
- ⬜ No auth between Open WebUI and sidecar (local network trust)

## Development

### Running tests against the sidecar

```bash
# Start sidecar
python -m perception_api.main

# Health check
curl http://localhost:8200/health

# Check backends
curl http://localhost:8200/backends

# Stage a test image
curl -X POST http://localhost:8200/attachments/stage \
  -F "session_id=test" \
  -F "turn_id=turn1" \
  -F "logical_name=test.jpg" \
  -F "file=@/path/to/test.jpg"

# List attachments
curl -X POST http://localhost:8200/attachments/list \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "turn_id": "turn1"}'

# Inspect image
curl -X POST http://localhost:8200/inspect \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "turn_id": "turn1", "logical_name": "test.jpg", "intent": "general"}'
```

## License

PoC / Internal use.
