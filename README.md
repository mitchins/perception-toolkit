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
│  │ Attachments  │  │  Florence-2   │  │  WD-14        │  │
│  │ Sandbox      │  │  Backend      │  │  Tagger       │  │
│  │              │  │              │  │  (optional)    │  │
│  │ per-session  │  │  general     │  │               │  │
│  │ per-turn     │  │  ocr         │  │  ranked tags  │  │
│  │ manifest     │  │  regions     │  │  with scores  │  │
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
| Model inference (Florence-2, WD-14) | Sidecar only |
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

Set the environment variable in Open WebUI:
```
PERCEPTION_SIDECAR_URL=http://localhost:8200
```

**3. Configure a model with the tools:**

In Open WebUI, assign the `perception_tools` toolset and `perception_filter` filter to your model (e.g. GPT-OSS).

### Benchmark Env (Local Model Comparison)

For local caption-quality benchmarking on Apple Silicon, use the dedicated Conda env:

```bash
conda env create -f environment-benchmark.yml
conda activate perception-bench-py312
```

This pins a clean Python 3.12 + `transformers` stack for Florence, Moondream, and the benchmark harness, instead of relying on whichever local env currently happens to work.

Current benchmark conclusions and backend decisions are documented in [benchmarks/RESEARCH_FINDINGS.md](benchmarks/RESEARCH_FINDINGS.md).

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
7. **Model calls** `inspect_image("image_1.jpg", intent="general")` → tool hits sidecar `/inspect` → Florence-2 runs inference → returns text description.
8. **Model answers** the user's question using the description.

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

| Intent | Description | Florence-2 Task |
|---|---|---|
| `general` | Detailed image description | `<MORE_DETAILED_CAPTION>` |
| `ocr` | Text extraction | `<OCR>` |
| `regions` | Region-level descriptions | `<DENSE_REGION_CAPTION>` |

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
