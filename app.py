"""Flask application for Drone Vision AI - Perplexity API + YOLO with Bounding Boxes"""
import os
import json
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from config import config
from utils import PerplexityAnalyzer, YOLODetector, get_image_info
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Load configuration
config_name = os.environ.get('FLASK_ENV', 'development')
app.config.from_object(config.get(config_name, config['default']))

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

print("=" * 70)
print("Initializing Drone Vision AI (Perplexity + YOLO)")
print("=" * 70)

# Initialize vision models
print("\n📊 Initializing Perplexity Analyzer...")
perplexity_api_key = os.environ.get("PERPLEXITY_API_KEY")
vision_analyzer = PerplexityAnalyzer(
    api_key=perplexity_api_key,
    demo_mode=app.config['DEMO_MODE']
)
print("✅ Perplexity Analyzer initialized")

print("\n🎯 Initializing YOLO Detector (with bounding boxes)...")
yolo_detector = YOLODetector(
    model_name=app.config['YOLO_MODEL'],
    demo_mode=app.config['DEMO_MODE']
)
print("✅ YOLO Detector initialized")

print("\n" + "=" * 70)
print("Application ready!")
print("=" * 70 + "\n")


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Analyze uploaded image with Perplexity Vision
    
    Returns JSON with terrain type, objects, and description
    """
    # Validate file upload
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Use JPG, PNG, GIF, or BMP'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get analysis mode
        mode = request.form.get('mode', 'perplexity')
        
        print(f"\n🔍 Analyzing image: {filename}")
        print(f"   Mode: {mode}")
        
        # Analyze image with Perplexity
        result = vision_analyzer.analyze_image(filepath)
        
        if result.get('success') or not result.get('success'):
            print(f"✅ Analysis complete - Source: {result.get('source')}")
            print(f"   Terrain: {result.get('terrain_type')}")
            print(f"   Objects: {', '.join(result.get('objects', []))}")
            
            return jsonify({
                'description': result.get('description', ''),
                'terrain_type': result.get('terrain_type', 'Unknown'),
                'objects': result.get('objects', []),
                'mode': mode,
                'file': filename,
                'timestamp': datetime.now().isoformat()
            })
        else:
            print(f"❌ Analysis failed: {result.get('description')}")
            return jsonify({
                'error': result.get('description', 'Unknown error'),
                'mode': mode
            }), 500
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect', methods=['POST'])
def detect():
    """
    Detect objects in uploaded image using YOLO with bounding boxes
    
    Returns JSON with detected objects and visualization
    """
    # Validate file upload
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Use JPG, PNG, GIF, or BMP'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get analysis mode
        mode = request.form.get('mode', 'hybrid')
        
        print(f"\n🎯 Detecting objects: {filename}")
        print(f"   Mode: {mode}")
        
        # Run detection using standard YOLO predict API with bounding boxes
        detection_result = yolo_detector.detect(
            image_path=filepath,
            conf_threshold=0.5,
            imgsz=640
        )
        
        if detection_result.get('success'):
            detections = detection_result.get('detections', [])
            print(f"✅ Detection complete - Found {detection_result.get('total_classes')} classes, {detection_result.get('total_objects')} objects")
            print(f"   Model: {detection_result.get('model_used')}")
            
            # Log detected classes
            for det in detections:
                print(f"   • {det['class']}: {det['count']} objects (confidence: {det['confidence']:.2f})")
            
            # Format response with visualization path
            response_data = {
                'detections': detections,
                'mode': mode,
                'source': detection_result.get('source'),
                'model_used': detection_result.get('model_used'),
                'total_objects': detection_result.get('total_objects'),
                'total_classes': detection_result.get('total_classes'),
                'visualization': detection_result.get('visualization'),
                'file': filename,
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify(response_data)
        else:
            error_msg = detection_result.get('error', 'Detection failed')
            print(f"❌ Detection failed: {error_msg}")
            return jsonify({
                'error': error_msg,
                'mode': mode
            }), 500
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/visualization/<path:fname>', methods=['GET'])
def get_visualization(fname):
    """
    Serve visualization image with bounding boxes
    
    Args:
        fname: Name of the visualization file
    """
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        if os.path.exists(filepath):
            return send_file(filepath, mimetype='image/jpeg')
        else:
            return jsonify({'error': 'Visualization not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/image-info', methods=['POST'])
def image_info():
    """Get image metadata"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        info = get_image_info(filepath)
        
        return jsonify({
            'info': info,
            'file': filename
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Check API status and model availability"""
    yolo_status = 'ready' if yolo_detector.model is not None else 'not_loaded'
    yolo_model = yolo_detector.model_name if yolo_detector.model is not None else 'none'
    perplexity_status = 'ready' if vision_analyzer.client is not None else 'demo_mode'
    
    return jsonify({
        'status': 'online',
        'perplexity_analyzer': perplexity_status,
        'yolo_detector': yolo_status,
        'yolo_model': yolo_model,
        'demo_mode': app.config['DEMO_MODE'],
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024),
        'timestamp': datetime.now().isoformat()
    })


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'error': f'File too large. Maximum size: {app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)}MB'
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Create uploads directory
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Run development server
    host = '0.0.0.0'
    port = int(os.environ.get('PORT', 5000))
    debug = app.config.get('DEBUG', False)
    
    print(f"\n🚀 Starting Flask server on {host}:{port}")
    print(f"   Debug mode: {debug}")
    print(f"   Demo mode: {app.config['DEMO_MODE']}")
    print(f"   Navigate to: http://localhost:{port}\n")
    
    app.run(
        host=host,
        port=port,
        debug=debug
    )
