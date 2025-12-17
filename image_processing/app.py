"""
Image Processing API Application.

A Flask-based REST API for various image processing operations including:
- Blur filtering (Gaussian, Median, Bilateral, etc.)
- Denoising (Non-Local Means, Adaptive, etc.)
- Colorization (via DeepAI API)
- Super Resolution (via DeepAI API)
- Background Removal (via DeepAI API)

Usage:
    python app.py

The API will start on http://localhost:5000
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify
from flask_cors import CORS

from config.settings import settings
from controllers.image_controller import image_bp


def create_app():
    """Create and configure Flask application."""
    app = Flask(__name__)
    
    # Configuration
    app.config["MAX_CONTENT_LENGTH"] = settings.MAX_CONTENT_LENGTH
    app.config["DEBUG"] = settings.DEBUG
    
    # Enable CORS
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(image_bp)
    
    # Root endpoint
    @app.route("/")
    def index():
        return jsonify({
            "name": "Image Processing API",
            "version": "1.0.0",
            "description": "REST API for AI-powered image processing",
            "endpoints": {
                "health": "/api/v1/images/health",
                "blur": "/api/v1/images/blur",
                "denoise": "/api/v1/images/denoise",
                "colorize": "/api/v1/images/colorize",
                "super_resolution": "/api/v1/images/super-resolution",
                "remove_background": "/api/v1/images/remove-background",
                "get_output": "/api/v1/images/output/<filename>",
                "options": "/api/v1/images/options/<operation>"
            },
            "documentation": "See README.md for detailed API documentation"
        })
    
    # Error handlers
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({"error": "Bad request", "message": str(error)}), 400
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Not found", "message": str(error)}), 404
    
    @app.errorhandler(413)
    def file_too_large(error):
        return jsonify({
            "error": "File too large",
            "message": f"Maximum file size is {settings.MAX_CONTENT_LENGTH / (1024*1024):.1f}MB"
        }), 413
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"error": "Internal server error", "message": str(error)}), 500
    
    return app


if __name__ == "__main__":
    app = create_app()
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║              Image Processing API v1.0.0                      ║
╠═══════════════════════════════════════════════════════════════╣
║  Server running on: http://{settings.HOST}:{settings.PORT}                     ║
║  Debug mode: {str(settings.DEBUG).lower()}                                          ║
╠═══════════════════════════════════════════════════════════════╣
║  Endpoints:                                                   ║
║    POST /api/v1/images/blur         - Apply blur filters      ║
║    POST /api/v1/images/denoise      - Denoise images          ║
║    POST /api/v1/images/colorize     - Colorize B&W images     ║
║    POST /api/v1/images/super-resolution - Upscale images      ║
║    POST /api/v1/images/remove-background - Remove backgrounds ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    app.run(
        host=settings.HOST,
        port=settings.PORT,
        debug=settings.DEBUG
    )
