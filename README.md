# 3D Generation Pipeline - Technical Documentation

## Overview

This project is an automated AI-powered pipeline for generating 3D models from 2D images. It transforms user-provided images into high-quality 3D Gaussian Splatting representations saved as PLY (Polygon File Format) files. The pipeline uses a multi-stage approach combining image editing, background removal, and 3D generation through state-of-the-art machine learning models.

## Architecture

### System Components

The pipeline consists of three primary processing modules orchestrated by a central `GenerationPipeline`:

1. **Image Editing Module** (Qwen-Image-Edit-2509)
2. **Background Removal Module** (RMBG 2.0)
3. **3D Generation Module** (Trellis Image-to-3D)

### Tech Stack

- **Framework**: FastAPI (REST API server)
- **ML Framework**: PyTorch with CUDA support
- **Image Processing**: PIL, torchvision
- **3D Generation**: Trellis (custom sparse structure VAE)
- **Compression**: pyspz (SPZ format compression)
- **Models**: Hugging Face Transformers, Diffusers

---

## Core Models

### 1. Qwen Image Edit Model

**Location**: [`modules/image_edit/qwen_edit_module.py`](pipeline_service/modules/image_edit/qwen_edit_module.py)

**Model ID**: `Qwen/Qwen-Image-Edit-2509`

**Purpose**: Pre-processes input images to optimize them for 3D generation by:
- Converting images to three-quarters view
- Removing watermarks and background details
- Ensuring the object is fully visible
- Creating neutral solid color backgrounds
- Sharpening image details
- Standardizing image quality

**Architecture**:
- Uses `QwenImageEditPlusPipeline` from Hugging Face Diffusers
- Employs `QwenImageTransformer2DModel` transformer architecture
- Supports both text prompts and embedded prompts (via safetensors)
- Uses flow-based diffusion sampling with configurable steps

**Key Parameters**:
- `num_inference_steps`: 8 (lightning-fast distilled model)
- `true_cfg_scale`: 1.0
- `height/width`: 1024x1024
- Data type: bfloat16 for efficiency

**Prompts** ([`config/qwen_edit_prompt.json`](pipeline_service/config/qwen_edit_prompt.json)):
- **Positive**: "Show this object in three-quarters view and make sure it is fully visible. Turn background neutral solid color contrasting with an object..."
- **Negative**: Excludes NSFW content, low quality, multiple objects, complex backgrounds, artistic styles, etc.

---

### 2. Background Removal Model

**Location**: [`modules/background_removal/rmbg_manager.py`](pipeline_service/modules/background_removal/rmbg_manager.py)

**Model ID**: `hiepnd11/rm_back2.0`

**Purpose**: Isolates the primary object by removing the background and creating an alpha-masked RGBA image.

**Process**:
1. Checks if image already has alpha channel - if yes, skips processing
2. Converts image to RGB and resizes to 1024x1024
3. Uses segmentation model to generate alpha mask
4. Detects bounding box around the object (threshold: 0.8)
5. Centers and crops the object with configurable padding
6. Resizes output to 518x518 (optimized for Trellis)

**Key Parameters**:
- `input_image_size`: (1024, 1024) - processing resolution
- `output_image_size`: (518, 518) - Trellis-optimized output
- `padding_percentage`: 0.2 (20% padding around object)
- `limit_padding`: true - prevents padding from exceeding image bounds

**Technical Details**:
- Uses `AutoModelForImageSegmentation`
- Normalization: ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Mask quantization at 0.8 threshold for clean edges
- Automatic centering and smart cropping

---

### 3. Trellis 3D Generation Model

**Location**: [`modules/gs_generator/trellis_manager.py`](pipeline_service/modules/gs_generator/trellis_manager.py)

**Model ID**: `jetx/trellis-image-large`

**Purpose**: Converts 2D images into 3D Gaussian Splatting representations.

**Architecture** ([`libs/trellis/pipelines/trellis_image_to_3d.py`](pipeline_service/libs/trellis/pipelines/trellis_image_to_3d.py)):

Trellis uses a two-stage diffusion process with sparse structured latents:

#### Stage 1: Sparse Structure Generation
- **Model**: Sparse Structure Flow Model
- **Purpose**: Generates a sparse voxel grid defining object occupancy
- **Input**: Random noise + image conditioning (DINOv2 features)
- **Output**: 3D coordinates of occupied voxels
- **Resolution**: Configurable voxel grid
- **Parameters**:
  - `sparse_structure_steps`: 8 sampling steps
  - `sparse_structure_cfg_strength`: 5.75 (classifier-free guidance)

#### Stage 2: Structured Latent Sampling (SLAT)
- **Model**: SLAT Flow Model
- **Purpose**: Generates detailed features at each occupied voxel
- **Input**: Sparse coordinates + image conditioning
- **Output**: Feature-rich sparse tensor
- **Parameters**:
  - `slat_steps`: 20 sampling steps
  - `slat_cfg_strength`: 2.4 (classifier-free guidance)

#### Image Conditioning
- Uses **DINOv2** (`facebookresearch/dinov2`) for image encoding
- Extracts prenormalized patch tokens as conditioning
- Supports both positive conditioning and negative (zero) conditioning

#### Decoding to Gaussian Splatting
- **Decoder**: SLAT Gaussian Decoder ([`libs/trellis/models/structured_latent_vae/decoder_gs.py`](pipeline_service/libs/trellis/models/structured_latent_vae/decoder_gs.py))
- Converts sparse latent tensors to Gaussian splats
- Outputs 3D Gaussians with position, color, opacity, and covariance
- Saved in PLY format compatible with standard viewers

**Key Features**:
- `num_oversamples`: 3 - generates multiple candidates, selects smallest/best
- Sparse representation for efficiency
- Flow-matching diffusion for high quality
- Native CUDA acceleration with flash attention

---

## Pipeline Workflow

### Entry Point: [`serve.py`](pipeline_service/serve.py)

The FastAPI application provides three main endpoints for 3D generation:

#### 1. `/generate` - Binary PLY Upload
```bash
curl -X POST "http://localhost:10006/generate" \
  -F "prompt_image_file=@image.png" \
  -F "seed=42" \
  -o model.ply
```

#### 2. `/generate-spz` - Compressed SPZ Upload
```bash
curl -X POST "http://localhost:10006/generate-spz" \
  -F "prompt_image_file=@image.png" \
  -F "seed=42" \
  -o model.spz
```

#### 3. `/generate_from_base64` - JSON API
```bash
curl -X POST "http://localhost:10006/generate_from_base64" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_type": "image",
    "prompt_image": "<base64_encoded_image>",
    "seed": 42,
    "trellis_params": {
      "sparse_structure_steps": 8,
      "slat_steps": 20
    }
  }'
```

### Processing Pipeline

**Location**: [`modules/pipeline.py`](pipeline_service/modules/pipeline.py:103-169)

The `GenerationPipeline.generate_gs()` method orchestrates the complete workflow:

```
Input Image (base64 or bytes)
    ↓
1. Decode Image
    ↓ (PIL.Image)
2. Qwen Image Edit
    ↓ (edited Image, 1024x1024)
3. Background Removal
    ↓ (RGBA Image, 518x518)
4. Trellis 3D Generation
    ↓ (Gaussian Splatting)
5. PLY Export
    ↓
Output: PLY bytes (optionally compressed to SPZ)
```

#### Detailed Steps:

**Step 1: Seed Initialization** ([`pipeline.py:117-121`](pipeline_service/modules/pipeline.py#L117-L121))
- If `seed < 0`: generates random seed (0-10000)
- Sets random seed for reproducibility across all modules

**Step 2: Image Decoding** ([`pipeline.py:124`](pipeline_service/modules/pipeline.py#L124))
- Decodes base64 string to PIL Image
- Supports JPEG, PNG, WebP formats

**Step 3: Qwen Image Editing** ([`pipeline.py:127`](pipeline_service/modules/pipeline.py#L127))
- Enhances image for 3D generation
- Applies view transformation and cleanup
- Output: 1024x1024 RGB image

**Step 4: Background Removal** ([`pipeline.py:130`](pipeline_service/modules/pipeline.py#L130))
- Segments and isolates main object
- Creates alpha mask
- Centers and crops with padding
- Output: 518x518 RGBA image

**Step 5: 3D Generation** ([`pipeline.py:138-144`](pipeline_service/modules/pipeline.py#L138-L144))
- Generates sparse structure
- Samples structured latents
- Decodes to Gaussian Splatting
- Output: PLY file as bytes

**Step 6: Optional Compression** ([`serve.py:65-69`](pipeline_service/serve.py#L65-L69))
- If enabled: compresses PLY to SPZ format using pyspz
- Significantly reduces file size for transmission

**Step 7: Response** ([`pipeline.py:163-169`](pipeline_service/modules/pipeline.py#L163-L169))
- Returns generation time
- Returns PLY bytes (base64 encoded if JSON endpoint)
- Optionally includes intermediate images

---

## Configuration

### Settings ([`config/settings.py`](pipeline_service/config/settings.py))

All configuration is managed through Pydantic settings with environment variable support:

#### API Settings
- `host`: "0.0.0.0"
- `port`: 10006

#### GPU Settings
- `qwen_gpu`: 0 - GPU for Qwen and Background Removal
- `trellis_gpu`: 0 - GPU for Trellis
- `dtype`: "bf16" - Model precision (bfloat16)

#### Output Settings
- `save_generated_files`: false - Save outputs to disk
- `send_generated_files`: false - Include intermediate images in response
- `output_dir`: "generated_outputs" - Output directory
- `compression`: true - Enable SPZ compression

#### Trellis Settings
- `trellis_model_id`: "jetx/trellis-image-large"
- `trellis_sparse_structure_steps`: 8
- `trellis_sparse_structure_cfg_strength`: 5.75
- `trellis_slat_steps`: 20
- `trellis_slat_cfg_strength`: 2.4
- `trellis_num_oversamples`: 3

#### Qwen Edit Settings
- `qwen_edit_model_path`: "Qwen/Qwen-Image-Edit-2509"
- `qwen_edit_height`: 1024
- `qwen_edit_width`: 1024
- `num_inference_steps`: 8
- `true_cfg_scale`: 1.0

#### Background Removal Settings
- `background_removal_model_id`: "hiepnd11/rm_back2.0"
- `input_image_size`: (1024, 1024)
- `output_image_size`: (518, 518)
- `padding_percentage`: 0.2

---

## Data Flow & Schemas

### Request Schema ([`schemas/requests.py`](pipeline_service/schemas/requests.py))

```python
class GenerateRequest(BaseModel):
    prompt_type: Literal["text", "image"] = "image"
    prompt_image: str  # base64 encoded
    seed: int = -1  # -1 = random
    trellis_params: Optional[TrellisParamsOverrides] = None
```

### Response Schema ([`schemas/responses.py`](pipeline_service/schemas/responses.py))

```python
class GenerateResponse(BaseModel):
    generation_time: float
    ply_file_base64: Optional[str | bytes] = None
    image_edited_file_base64: Optional[str] = None
    image_without_background_file_base64: Optional[str] = None
```

### Trellis Parameters ([`schemas/trellis_schemas.py`](pipeline_service/schemas/trellis_schemas.py))

```python
class TrellisParams(OverridableModel):
    sparse_structure_steps: int
    sparse_structure_cfg_strength: float
    slat_steps: int
    slat_cfg_strength: float
    num_oversamples: int = 1
```

---

## Startup Sequence

### Initialization ([`pipeline.py:32-45`](pipeline_service/modules/pipeline.py#L32-L45))

1. **Load Qwen Edit Model**
   - Loads transformer and scheduler
   - Moves to GPU (qwen_gpu)
   - Prepares prompt embeddings

2. **Load Background Removal Model**
   - Downloads from Hugging Face if needed
   - Loads segmentation model
   - Moves to GPU (qwen_gpu)

3. **Load Trellis Pipeline**
   - Loads sparse structure flow model
   - Loads SLAT flow model
   - Loads Gaussian decoder
   - Loads DINOv2 for image conditioning
   - Moves to GPU (trellis_gpu)
   - Sets attention backend to "flash-attn"

4. **Warmup Generation** ([`pipeline.py:65-72`](pipeline_service/modules/pipeline.py#L65-L72))
   - Generates dummy 64x64 image
   - Runs complete pipeline with seed=42
   - Ensures all models are loaded and optimized
   - Clears GPU cache after warmup

5. **Ready to Serve**
   - API endpoints become available
   - Health check at `/health` returns "ready"

---

## GPU Memory Management

The pipeline implements automatic GPU memory management:

### Memory Cleanup ([`pipeline.py:58-63`](pipeline_service/modules/pipeline.py#L58-L63))

After each generation:
```python
def _clean_gpu_memory(self):
    gc.collect()
    torch.cuda.empty_cache()
```

### VRAM Requirements

Estimated VRAM usage:
- **Qwen Edit**: ~15GB
- **Background Removal**: ~3GB
- **Trellis**: ~40-60GB (depending on output complexity)
- **Total Recommended**: 80GB+ (e.g., A100 80GB)

With 2 GPUs, you can split:
- GPU 0: Qwen + RMBG (~18GB)
- GPU 1: Trellis (~50GB)

---

## Output Formats

### PLY (Polygon File Format)

Standard 3D Gaussian Splatting format containing:
- Vertex positions (x, y, z)
- Gaussian parameters (scale, rotation, opacity)
- Color information (RGB or spherical harmonics)
- Compatible with viewers like Polycam, Luma AI, etc.

### SPZ (Compressed Format)

Compressed version of PLY using pyspz:
- Significantly smaller file size
- Lossless compression
- Requires decompression before viewing
- Useful for network transmission

---

## Performance Characteristics

### Typical Generation Times

On A100 80GB GPU:
- **Qwen Edit**: ~2-3 seconds
- **Background Removal**: ~0.5-1 second
- **Trellis Generation**: ~8-15 seconds
- **Total Pipeline**: ~10-20 seconds per image

### Optimization Settings

For faster generation (lower quality):
- Reduce `trellis_sparse_structure_steps` to 4-6
- Reduce `trellis_slat_steps` to 12-15
- Reduce `trellis_num_oversamples` to 1

For higher quality (slower):
- Increase `trellis_sparse_structure_steps` to 12-16
- Increase `trellis_slat_steps` to 30-40
- Increase `trellis_num_oversamples` to 5-8

---

## Advanced Features

### Seed Control

The pipeline supports deterministic generation:
- `seed >= 0`: Reproducible results
- `seed = -1`: Random generation
- Seed affects all three stages consistently

### Parameter Overrides

Clients can override Trellis parameters per request:
```json
{
  "trellis_params": {
    "sparse_structure_steps": 12,
    "slat_cfg_strength": 3.0
  }
}
```

### Multi-Sampling

Trellis generates multiple candidates and selects the best:
- `num_oversamples` > 1: Generate N sparse structures
- Selects smallest structures (less noisy)
- Improves quality at cost of computation

---

## Docker Deployment

### Requirements
- NVIDIA GPU with CUDA 12.x
- Docker with NVIDIA Container Toolkit
- 80GB+ VRAM recommended

### Build & Run

```bash
# Build image
docker build -f docker/Dockerfile -t forge3d-pipeline:latest .

# Run with docker-compose
cd docker
docker-compose up -d

# Run with docker directly
docker run --gpus all -p 10006:10006 --env-file .env forge3d-pipeline:latest
```

---

## Error Handling

### Automatic Fallbacks

**Background Removal** ([`rmbg_manager.py:76-112`](pipeline_service/modules/background_removal/rmbg_manager.py#L76-L112)):
- If image already has alpha channel, skips processing
- If segmentation fails, returns original image
- Logs errors but doesn't crash pipeline

### Exception Handling

All endpoints wrap processing in try-catch:
- Logs detailed exception traces
- Returns HTTP 500 with error details
- Maintains service availability

---

## Key Technical Innovations

### 1. Sparse Structured Latents (SLAT)
- Efficiently represents 3D objects in sparse voxel space
- Only processes occupied voxels, not entire grid
- Dramatically reduces memory and computation

### 2. Two-Stage Flow Matching
- First stage: Coarse structure (occupancy)
- Second stage: Fine details (features)
- Enables high-quality generation with manageable compute

### 3. DINOv2 Conditioning
- Self-supervised vision model for image understanding
- Provides rich semantic features for 3D generation
- No fine-tuning required

### 4. Gaussian Splatting Output
- Modern 3D representation (vs mesh or NeRF)
- Supports view-dependent effects
- Fast rendering in real-time viewers

---

## File Structure

```
pipeline_service/
├── serve.py                 # FastAPI application entry point
├── config/
│   ├── settings.py          # Configuration management
│   └── qwen_edit_prompt.json # Qwen editing prompts
├── modules/
│   ├── pipeline.py          # Main orchestration pipeline
│   ├── image_edit/
│   │   ├── qwen_edit_module.py    # Qwen image editing
│   │   └── qwen_manager.py        # Qwen model management
│   ├── background_removal/
│   │   └── rmbg_manager.py        # Background removal service
│   └── gs_generator/
│       └── trellis_manager.py     # Trellis 3D generation
├── schemas/
│   ├── requests.py          # Request data models
│   ├── responses.py         # Response data models
│   └── trellis_schemas.py   # Trellis-specific schemas
└── libs/
    └── trellis/             # Trellis model implementation
        ├── pipelines/
        │   └── trellis_image_to_3d.py  # Main pipeline
        ├── models/          # VAE encoders/decoders
        └── modules/         # Neural network modules
```

---

## Conclusion

This pipeline represents a state-of-the-art approach to image-to-3D generation, combining:
- **Qwen Image Edit** for intelligent image preprocessing
- **RMBG 2.0** for precise background removal
- **Trellis** for cutting-edge sparse 3D generation

The modular design allows for easy customization, while the FastAPI interface provides flexible integration options. The pipeline is optimized for both quality and performance, supporting production deployments on modern GPU infrastructure.
