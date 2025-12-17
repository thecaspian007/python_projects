# Image Processing API

A Flask-based REST API for AI-powered image processing operations. This API provides endpoints for various image manipulation tasks including filtering, enhancement, and AI-based transformations.

## Features

- **Local Processing**: Blur, denoise using OpenCV
- **AI-Powered**: Colorization, super resolution, background removal via DeepAI API

## Installation

### Prerequisites

- Python 3.9+
- pip package manager
- DeepAI API key (free tier available at [deepai.org](https://deepai.org))

### Setup

1. **Clone and navigate to directory**:
   ```bash
   cd image_processing
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set DeepAI API key**:
   ```bash
   export DEEPAI_API_KEY="your-api-key-here"
   ```

5. **Run the application**:
   ```bash
   python app.py
   ```

The API will start on `http://localhost:5000`

## API Endpoints

### Health Check
```
GET /api/v1/images/health
```

### Blur Image
```
POST /api/v1/images/blur
```

Apply blur filter to an image.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| image | File | - | Image file to process |
| image_url | String | - | URL of image (alternative to file) |
| method | String | gaussian | Blur method: gaussian, median, bilateral, box, motion |
| kernel_size | Integer | 5 | Size of blur kernel (must be odd) |
| sigma | Float | 0 | Sigma for Gaussian blur |

**Example**:
```bash
curl -X POST http://localhost:5000/api/v1/images/blur \
  -F "image=@photo.jpg" \
  -F "method=gaussian" \
  -F "kernel_size=5"
```

### Denoise Image
```
POST /api/v1/images/denoise
```

Remove noise from an image.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| image | File | - | Image file to process |
| method | String | nlm_color | Method: nlm, nlm_color, bilateral, morphological, adaptive |
| h | Float | 10 | Filter strength (higher = more noise removal) |

**Example**:
```bash
curl -X POST http://localhost:5000/api/v1/images/denoise \
  -F "image=@noisy_image.jpg" \
  -F "method=nlm_color" \
  -F "h=10"
```

### Colorize Image (DeepAI)
```
POST /api/v1/images/colorize
```

Add color to black and white images using AI.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| image | File | - | B&W image file to colorize |
| save_local | Boolean | true | Save result locally |

**Example**:
```bash
curl -X POST http://localhost:5000/api/v1/images/colorize \
  -F "image=@bw_photo.jpg"
```

### Super Resolution (DeepAI)
```
POST /api/v1/images/super-resolution
```

Upscale images using AI while maintaining quality.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| image | File | - | Image file to upscale |
| save_local | Boolean | true | Save result locally |

**Example**:
```bash
curl -X POST http://localhost:5000/api/v1/images/super-resolution \
  -F "image=@small_image.jpg"
```

### Remove Background (DeepAI)
```
POST /api/v1/images/remove-background
```

Automatically remove background from images.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| image | File | - | Image file |
| save_local | Boolean | true | Save result locally |

**Example**:
```bash
curl -X POST http://localhost:5000/api/v1/images/remove-background \
  -F "image=@portrait.jpg"
```

## Image Processing Models Used

### 1. Blur Filters (OpenCV)

| Method | Description | Best For |
|--------|-------------|----------|
| **Gaussian Blur** | Smooths using Gaussian kernel | General smoothing, noise reduction |
| **Median Blur** | Replaces pixel with median of neighbors | Salt-and-pepper noise |
| **Bilateral Filter** | Edge-preserving smoothing | Preserving edges while smoothing |
| **Box Blur** | Simple averaging filter | Fast, basic smoothing |
| **Motion Blur** | Directional blur effect | Simulating motion |

### 2. Denoising Methods (OpenCV)

| Method | Description | Best For |
|--------|-------------|----------|
| **NLM (Non-Local Means)** | Compares patches across image | High-quality denoising |
| **NLM Colored** | NLM for color images | Color image denoising |
| **Bilateral** | Edge-preserving denoising | Preserving sharp edges |
| **Morphological** | Opening/closing operations | Binary and grayscale noise |
| **Adaptive** | Adaptive thresholding | Document cleanup |

### 3. DeepAI API Models

| Endpoint | Model | Description |
|----------|-------|-------------|
| **Colorizer** | Neural Colorization | Adds realistic colors to grayscale images using deep learning |
| **torch-srgan** | SRGAN | Super Resolution GAN for 4x upscaling with detail preservation |
| **background-remover** | Semantic Segmentation | Detects and removes backgrounds using AI segmentation |

## Project Structure

```
image_processing/
├── app.py                      # Flask application entry point
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── config/
│   └── settings.py             # Configuration
├── controllers/
│   └── image_controller.py     # REST API endpoints
├── services/
│   ├── base_processor.py       # Abstract base class
│   ├── filtering/
│   │   ├── blur_service.py     # Blur operations
│   │   └── denoise_service.py  # Denoising operations
│   ├── enhancement/
│   │   ├── colorization_service.py  # AI colorization
│   │   └── super_resolution_service.py  # AI upscaling
│   └── detection/
│       └── background_removal_service.py  # AI background removal
├── models/
│   └── image_result.py         # Result data models
├── utils/
│   └── image_utils.py          # Image I/O utilities
└── external_apis/
    ├── base_client.py          # Abstract API client
    └── deepai_client.py        # DeepAI integration
```

## Response Format

All endpoints return JSON with the following structure:

```json
{
  "status": "success",
  "operation": "blur",
  "message": "Successfully applied gaussian blur",
  "output_url": "https://...",      // For AI operations
  "output_path": "output/abc123.png",
  "processing_time": 0.45,
  "metadata": {
    "method": "gaussian",
    "kernel_size": 5
  },
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DEEPAI_API_KEY` | DeepAI API key for AI features | Yes (for AI endpoints) |

## License

MIT License
