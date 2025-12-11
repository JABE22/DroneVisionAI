# 🚁 Drone Vision AI

**Advanced drone image analysis combining Perplexity Vision AI and YOLO object detection**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![YOLO](https://img.shields.io/badge/YOLO-v8-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

### 🧠 **Perplexity Vision Analysis**
- Terrain type classification (urban, rural, forest, water, etc.)
- Object detection and listing
- Scene description generation
- Powered by Perplexity AI sonar-pro model

### 🎯 **YOLOv8 Object Detection**
- Real-time object detection
- Confidence score tracking
- Bounding box visualization
- Multi-class detection with color-coded boxes

### 🎨 **Modern Web Interface**
- Drag-and-drop image upload
- Live image preview
- Beautiful gradient UI
- Responsive design (mobile-friendly)
- Real-time result display
- Professional bounding box visualizations

---

## 🚀 Quick Start

### 1. Clone or Download Project

```bash
# Extract the project
unzip drone-vision-ai.zip
cd drone-vision-ai
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
python3 -c "from ultralytics import YOLO; YOLO('yolo11l.pt')"
```

### 4. Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit .env and add your Perplexity API key
export PERPLEXITY_API_KEY='your-api-key'

# Disable demo mode
export DEMO_MODE=false
```

### 5. Run Application

```bash
python3 app.py
```

Open browser: `http://localhost:5000`

---

## 📋 Configuration

### Environment Variables (.env)

```bash
# Required
PERPLEXITY_API_KEY=pplx-your-api-key-here

# Optional
FLASK_ENV=development         # development or production
DEMO_MODE=false               # true for demo mode (no API needed)
PORT=5000
YOLO_MODEL=yolo11l.pt         # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x, yolo11l, etc.
DEBUG=True                    # Flask debug mode
```

### Getting Perplexity API Key

1. Visit: https://www.perplexity.ai/
2. Sign up for an account
3. Go to Settings → API Keys
4. Create new API key
5. Copy and paste into `.env`

---

## 📁 Project Structure

```
drone-vision-ai/
├── app.py                 # Flask application (main entry point)
├── config.py              # Configuration classes
├── utils.py               # Vision models (Perplexity + YOLO)
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
├── templates/
│   └── index.html        # Web interface
│
├── uploads/              # Auto-created, stores images & visualizations
│
└── venv/                 # Virtual environment
```

---

## 🎯 Usage

### Text Analysis (Perplexity)

1. **Upload** drone image (JPG, PNG, GIF, BMP)
2. Click **"✨ Analyze Image"**
3. Wait 5-15 seconds
4. Get:
   - 🗺️ **Terrain Type** (urban, forest, etc.)
   - 🔍 **Objects** (buildings, roads, vehicles, etc.)
   - 📝 **Description** (1-2 sentence summary)

### Object Detection (YOLO)

1. **Upload** drone image
2. Click **"🎯 Detect Objects"**
3. Wait 1-2 seconds
4. Get:
   - 📊 Detection list with confidence scores
   - 📸 **Bounding box image** (colored rectangles around objects)
   - 📈 Statistics (total objects, classes, model used)

---

## 🔧 API Endpoints

### Analyze Image (Perplexity)
```bash
POST /api/analyze
Content-Type: multipart/form-data

Response:
{
  "terrain_type": "Urban Area",
  "objects": ["Buildings", "Roads", "Vehicles"],
  "description": "Aerial view of residential area...",
  "file": "20241209_150000_image.jpg",
  "timestamp": "2024-12-09T15:00:00"
}
```

### Detect Objects (YOLO)
```bash
POST /api/detect
Content-Type: multipart/form-data

Response:
{
  "detections": [
    {"class": "car", "confidence": 0.95, "count": 3},
    {"class": "building", "confidence": 0.92, "count": 8}
  ],
  "visualization": "20241209_150000_image_bbox.jpg",
  "total_objects": 11,
  "total_classes": 2,
  "model_used": "yolo11l.pt"
}
```

### Get Visualization
```bash
GET /api/visualization/{filename}
# Returns the bounding box image
```

### Check Status
```bash
GET /api/status

Response:
{
  "status": "online",
  "perplexity_analyzer": "ready",
  "yolo_detector": "ready",
  "yolo_model": "yolo11l.pt",
  "demo_mode": false
}
```

---

## 🧪 Testing

### Quick Test (Demo Mode)

```bash
export DEMO_MODE=true
python3 app.py
# No API key needed, uses mock data
```

### With Real API

```bash
export DEMO_MODE=false
export PERPLEXITY_API_KEY='your-key'
python3 app.py
```

### Manual API Test

```bash
# Start Flask, then in another terminal:

curl -X POST -F "file=@test.jpg" \
  http://localhost:5000/api/detect \
  -s | python3 -m json.tool
```

---

## 📊 Performance

### Speed (Approximate)

- **Perplexity Analysis**: 5-15 seconds (API-based)
- **YOLO Detection**: 1-2 seconds (GPU: <1s)
- **Bounding Box Rendering**: <1 second

### Resource Usage

- **Memory**: ~2-4 GB (for YOLO11l + Flask)
- **GPU**: Optional (faster with CUDA, works on CPU)
- **Storage**: ~100-500 MB for models

### Scalability

- Processes one image at a time
- Can handle simultaneous users (Flask multi-threaded)
- Recommended: Deploy with Gunicorn for production

---

## 🚀 Production Deployment

### Using Gunicorn

```bash
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 app:app
```

### Using Systemd (Ubuntu/Linux)

```bash
sudo tee /etc/systemd/system/drone-vision.service << EOF
[Unit]
Description=Drone Vision AI
After=network.target

[Service]
User=www-data
WorkingDirectory=/home/user/drone-vision-ai
Environment="PERPLEXITY_API_KEY=your-key"
Environment="FLASK_ENV=production"
ExecStart=/home/user/drone-vision-ai/venv/bin/gunicorn \
  -w 4 -b 127.0.0.1:5000 --timeout 120 app:app

Restart=always
StandardOutput=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable drone-vision
sudo systemctl start drone-vision
```

### Using Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python3 -c "from ultralytics import YOLO; YOLO('yolo11l.pt')"

COPY . .

ENV FLASK_ENV=production
EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

---

## 🛠️ Troubleshooting

### Issue: Perplexity API Key Not Found
```bash
export PERPLEXITY_API_KEY='pplx-your-key'
python3 app.py
```

### Issue: YOLO Model Not Loading
```bash
# Download manually
python3 -c "from ultralytics import YOLO; YOLO('yolo11l.pt')"

# Or use demo mode
export DEMO_MODE=true
```

### Issue: Bounding Boxes Not Appearing
```bash
# Ensure OpenCV installed
pip install --upgrade opencv-python

# Check file permissions
chmod 755 uploads/

# Check console for errors
python3 app.py 2>&1 | grep -i "error\|warning"
```

### Issue: "ModuleNotFoundError: No module named 'perplexity'"
```bash
pip install perplexityai
```

---

## 📚 Architecture

### Backend Stack
- **Flask** - Web framework
- **Perplexity API** - Vision analysis (sonar-pro model)
- **YOLO11l** - Object detection (ultralytics)
- **OpenCV** - Image processing
- **PIL/Pillow** - Image manipulation

### Frontend Stack
- **HTML5** - Structure
- **CSS3** - Styling (gradients, flexbox, animations)
- **Vanilla JavaScript** - No dependencies
- **Drag-drop API** - File upload

### Data Flow
```
User Upload
    ↓
Flask receives file
    ↓
├─ Analysis Path → Perplexity API → Terrain + Objects
└─ Detection Path → YOLO Model → Bounding Boxes
    ↓
Save visualization
    ↓
Return to Frontend
    ↓
Display results
```

---

## 🔐 Security

- ✅ CSRF protection (built-in Flask)
- ✅ File type validation
- ✅ File size limits (10 MB default)
- ✅ Secure filename handling
- ✅ API key in environment (not hardcoded)

### Recommendations
- Use HTTPS in production
- Change `SECRET_KEY` in config
- Set `DEBUG=False` in production
- Use rate limiting middleware
- Validate API keys server-side

---

## 📖 Documentation

### Key Files

| File | Purpose |
|------|---------|
| `app.py` | Flask routes and error handling |
| `config.py` | Environment-specific configuration |
| `utils.py` | PerplexityAnalyzer and YOLODetector classes |
| `templates/index.html` | Web interface |

### Class Documentation

#### `PerplexityAnalyzer`
```python
analyzer = PerplexityAnalyzer(api_key='...', demo_mode=False)
result = analyzer.analyze_image('path/to/image.jpg')
# Returns: {'terrain_type', 'objects', 'description', ...}
```

#### `YOLODetector`
```python
detector = YOLODetector(model_name='yolo11l.pt', demo_mode=False)
result = detector.detect('path/to/image.jpg', conf_threshold=0.5)
# Returns: {'detections', 'visualization', 'total_objects', ...}
```

---

## 🤝 Contributing

Found a bug? Have suggestions?

1. Test the issue thoroughly
2. Check the logs for error messages
3. Submit details with:
   - Error message
   - Steps to reproduce
   - System info (OS, Python version)
   - Image file (if applicable)

---

## 📜 License

MIT License - Feel free to use, modify, and distribute

---

## 📞 Support

- **Perplexity API Issues**: https://www.perplexity.ai/
- **YOLO11l Documentation**: https://docs.ultralytics.com/
- **Flask Help**: https://flask.palletsprojects.com/

---

## 🎉 Acknowledgments

- **Perplexity AI** - Vision analysis
- **Ultralytics** - YOLO11l (or you can try other versions) object detection
- **Flask** - Web framework
- **OpenCV** - Image processing

---

## ✨ What's Included

✅ Modern web interface  
✅ Perplexity Vision analysis  
✅ YOLO11l object detection  
✅ Bounding box visualization  
✅ RESTful API  
✅ Error handling  
✅ Demo mode  
✅ Production ready  

---

**Version**: 1.0.0  
**Last Updated**: December 10, 2025  
**Status**: ✅ Fully Functional

Start analyzing drone images now! 🚀
