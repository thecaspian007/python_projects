"""
Image Processing API Controller.

This module contains all REST API endpoints for image processing operations.
"""
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os

from config.settings import settings
from utils.image_utils import ImageUtils
from services.filtering.blur_service import BlurService
from services.filtering.denoise_service import DenoiseService
from services.enhancement.colorization_service import ColorizationService
from services.enhancement.super_resolution_service import SuperResolutionService
from services.detection.background_removal_service import BackgroundRemovalService


# Create Blueprint
image_bp = Blueprint("images", __name__, url_prefix="/api/v1/images")

# Initialize services
blur_service = BlurService()
denoise_service = DenoiseService()
colorization_service = ColorizationService()
super_resolution_service = SuperResolutionService()
background_removal_service = BackgroundRemovalService()


def get_image_from_request():
    """Extract image from request (file upload or URL)."""
    if "image" in request.files:
        file = request.files["image"]
        if file.filename == "":
            return None, "No file selected"
        if not ImageUtils.is_valid_extension(file.filename):
            return None, f"Invalid file type. Allowed: {settings.ALLOWED_EXTENSIONS}"
        
        image_bytes = file.read()
        image = ImageUtils.read_image_from_bytes(image_bytes)
        return image, None
    
    elif "image_url" in request.form or "image_url" in request.json if request.is_json else False:
        image_url = request.form.get("image_url") or (request.json.get("image_url") if request.is_json else None)
        if not image_url:
            return None, "No image URL provided"
        
        try:
            image = ImageUtils.download_image_from_url(image_url)
            return image, None
        except Exception as e:
            return None, f"Failed to download image: {str(e)}"
    
    return None, "No image provided. Use 'image' for file upload or 'image_url' for URL"


@image_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Image Processing API",
        "version": "1.0.0"
    })


@image_bp.route("/blur", methods=["POST"])
def blur_image():
    """
    Apply blur filter to an image.
    
    Form Data:
        image: Image file to process
        image_url: URL of image (alternative to file)
        method: Blur method (gaussian, median, bilateral, box, motion)
        kernel_size: Size of blur kernel (default: 5)
        sigma: Sigma for Gaussian blur (default: 0)
    
    Returns:
        JSON with processing result and output path
    """
    image, error = get_image_from_request()
    if error:
        return jsonify({"error": error}), 400
    
    method = request.form.get("method", "gaussian")
    kernel_size = int(request.form.get("kernel_size", 5))
    sigma = float(request.form.get("sigma", 0))
    
    result = blur_service.process(
        image,
        method=method,
        kernel_size=kernel_size,
        sigma=sigma
    )
    
    status_code = 200 if result.status.value == "success" else 500
    return jsonify(result.to_dict()), status_code


@image_bp.route("/denoise", methods=["POST"])
def denoise_image():
    """
    Apply denoising to an image.
    
    Form Data:
        image: Image file to process
        image_url: URL of image (alternative to file)
        method: Denoising method (nlm, nlm_color, bilateral, morphological, adaptive)
        h: Filter strength (default: 10)
    
    Returns:
        JSON with processing result and output path
    """
    image, error = get_image_from_request()
    if error:
        return jsonify({"error": error}), 400
    
    method = request.form.get("method", "nlm_color")
    h = float(request.form.get("h", 10))
    
    result = denoise_service.process(image, method=method, h=h)
    
    status_code = 200 if result.status.value == "success" else 500
    return jsonify(result.to_dict()), status_code


@image_bp.route("/colorize", methods=["POST"])
def colorize_image():
    """
    Colorize a black and white image using DeepAI API.
    
    Form Data:
        image: Image file to process
        image_url: URL of image (alternative to file)
        save_local: Whether to save result locally (default: true)
    
    Returns:
        JSON with processing result, output URL, and local path
    """
    image, error = get_image_from_request()
    if error:
        return jsonify({"error": error}), 400
    
    save_local = request.form.get("save_local", "true").lower() == "true"
    
    result = colorization_service.process(image, save_local=save_local)
    
    status_code = 200 if result.status.value == "success" else 500
    return jsonify(result.to_dict()), status_code


@image_bp.route("/super-resolution", methods=["POST"])
def super_resolution():
    """
    Upscale an image using AI super resolution via DeepAI API.
    
    Form Data:
        image: Image file to process
        image_url: URL of image (alternative to file)
        save_local: Whether to save result locally (default: true)
    
    Returns:
        JSON with processing result, output URL, and local path
    """
    image, error = get_image_from_request()
    if error:
        return jsonify({"error": error}), 400
    
    save_local = request.form.get("save_local", "true").lower() == "true"
    
    result = super_resolution_service.process(image, save_local=save_local)
    
    status_code = 200 if result.status.value == "success" else 500
    return jsonify(result.to_dict()), status_code


@image_bp.route("/remove-background", methods=["POST"])
def remove_background():
    """
    Remove background from an image using DeepAI API.
    
    Form Data:
        image: Image file to process
        image_url: URL of image (alternative to file)
        save_local: Whether to save result locally (default: true)
    
    Returns:
        JSON with processing result, output URL, and local path
    """
    image, error = get_image_from_request()
    if error:
        return jsonify({"error": error}), 400
    
    save_local = request.form.get("save_local", "true").lower() == "true"
    
    result = background_removal_service.process(image, save_local=save_local)
    
    status_code = 200 if result.status.value == "success" else 500
    return jsonify(result.to_dict()), status_code


@image_bp.route("/output/<filename>", methods=["GET"])
def get_output_image(filename):
    """
    Retrieve a processed output image.
    
    Args:
        filename: Name of the output file
    
    Returns:
        Image file
    """
    filepath = os.path.join(settings.OUTPUT_DIR, secure_filename(filename))
    
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    
    return send_file(filepath)


@image_bp.route("/options/<operation>", methods=["GET"])
def get_operation_options(operation):
    """
    Get supported options for a specific operation.
    
    Args:
        operation: Operation name (blur, denoise, colorize, super-resolution, remove-background)
    
    Returns:
        JSON with supported options
    """
    services = {
        "blur": blur_service,
        "denoise": denoise_service,
        "colorize": colorization_service,
        "super-resolution": super_resolution_service,
        "remove-background": background_removal_service
    }
    
    if operation not in services:
        return jsonify({
            "error": f"Unknown operation: {operation}",
            "available_operations": list(services.keys())
        }), 400
    
    return jsonify({
        "operation": operation,
        "options": services[operation].get_supported_options()
    })
