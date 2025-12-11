"""Vision model utilities - Perplexity API + YOLO with bounding boxes (Using YOLO.plot())"""
import os
import json
import base64
from typing import List, Dict, Optional
from PIL import Image
import cv2
import numpy as np

try:
    from perplexity import Perplexity
except ImportError:
    Perplexity = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class PerplexityAnalyzer:
    """Handle Perplexity API calls for drone image analysis"""
    
    def __init__(self, api_key: str = None, demo_mode: bool = False):
        """
        Initialize Perplexity analyzer
        
        Args:
            api_key: Perplexity API key
            demo_mode: If True, return mock results
        """
        self.api_key = api_key or os.environ.get('PERPLEXITY_API_KEY')
        self.demo_mode = demo_mode
        self.client = None
        
        if not demo_mode:
            try:
                if Perplexity is None:
                    raise ImportError("perplexity not installed. Install with: pip install perplexity-py")
                
                if not self.api_key:
                    raise ValueError("PERPLEXITY_API_KEY not set")
                
                self.client = Perplexity(api_key=self.api_key)
                print("✅ Perplexity API initialized")
            except Exception as e:
                print(f"⚠️ Warning: Could not initialize Perplexity: {str(e)}")
                self.client = None
    
    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyze drone image with Perplexity Vision (sonar-pro model)
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict with analysis result including terrain_type, objects, description
        """
        if self.demo_mode or self.client is None:
            return self._demo_analysis()
        
        try:
            # Convert image to base64 data URI
            data_uri = self._image_to_data_uri(image_path)
            
            # Prompt for drone image analysis
            prompt_template = """Проанализируй этот аэрофотоснимок и определи:
1. Тип местности (город/лес/поле/водоем/горы/промзона)
2. Наличие объектов (здания/дороги/транспорт/растительность)
3. Краткое описание (1-2 предложения)

Ответь СТРОГО в формате JSON:
{
  "terrain_type": "тип местности",
  "objects": ["объект1", "объект2"],
  "description": "краткое описание"
}"""
            
            # Call Perplexity API with vision
            completion = self.client.chat.completions.create(
                model="sonar-pro",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_template},
                            {"type": "image_url", "image_url": {"url": data_uri}}
                        ]
                    }
                ]
            )
            
            result_text = completion.choices[0].message.content
            
            # Extract JSON from response
            try:
                if '```json' in result_text:
                    json_str = result_text.split('```json')[1].split('```')[0]
                elif '```' in result_text:
                    json_str = result_text.split('```')[1].split('```')[0]
                else:
                    json_str = result_text
                
                result = json.loads(json_str.strip())
                return {
                    'terrain_type': result.get('terrain_type', 'N/A'),
                    'objects': result.get('objects', []),
                    'description': result.get('description', ''),
                    'success': True,
                    'source': 'perplexity'
                }
            except json.JSONDecodeError as jde:
                return {
                    'terrain_type': 'error_json',
                    'objects': [],
                    'description': result_text,
                    'success': False,
                    'source': 'perplexity',
                    'error': f'JSON parse error: {jde}'
                }
            
        except Exception as e:
            return {
                'terrain_type': 'error',
                'objects': [],
                'description': f'Error: {str(e)}',
                'success': False,
                'source': 'perplexity'
            }
    
    def _image_to_data_uri(self, image_path: str) -> str:
        """Convert image file to base64 data URI"""
        with open(image_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        ext = image_path.split('.')[-1].lower()
        mime = f"image/{'jpeg' if ext == 'jpg' else ext}"
        return f"data:{mime};base64,{b64}"
    
    def _demo_analysis(self) -> Dict:
        """Return mock analysis result for demo mode"""
        return {
            'terrain_type': 'Urban Area',
            'objects': ['Buildings', 'Roads', 'Vehicles', 'Trees'],
            'description': 'Aerial view of a dense urban area with mixed-use buildings, well-developed road network, and green spaces.',
            'success': True,
            'source': 'demo'
        }


class YOLODetector:
    """Handle YOLOv8 object detection with bounding box visualization using YOLO.plot()"""
    
    def __init__(self, model_name: str = 'yolov8s.pt', demo_mode: bool = False):
        """
        Initialize YOLO detector
        
        Args:
            model_name: Model to load (e.g., 'yolov8s.pt', 'yolo11n.pt')
            demo_mode: If True, return mock detections
        """
        self.model_name = model_name
        self.demo_mode = demo_mode
        self.model = None
        
        if not demo_mode:
            try:
                if YOLO is None:
                    raise ImportError("ultralytics not installed. Install with: pip install ultralytics")
                
                print(f"Loading YOLO model: {model_name}")
                self.model = YOLO(model_name)
                print(f"✅ YOLO model loaded: {model_name}")
            except Exception as e:
                print(f"⚠️ Warning: Could not load YOLO model: {str(e)}")
                self.model = None
    
    def detect(self, image_path: str, conf_threshold: float = 0.5, imgsz: int = 640) -> Dict:
        """
        Run YOLOv8 detection on image using standard predict API
        
        Args:
            image_path: Path to image file
            conf_threshold: Confidence threshold for detections (0-1)
            imgsz: Image size for inference (default 640)
            
        Returns:
            dict with detections, bounding boxes, and visualization
        """
        if self.demo_mode or self.model is None:
            return self._demo_detection(image_path)
        
        try:
            print(f"DEBUG: Starting YOLO detection on {image_path}")
            
            # Standard YOLO predict call
            results = self.model.predict(
                source=image_path,
                imgsz=imgsz,
                conf=conf_threshold,
                verbose=False
            )
            
            detections = []
            class_stats = {}
            
            print(f"DEBUG: YOLO returned {len(results)} result(s)")
            
            # Process results
            if results and len(results) > 0:
                result = results[0]
                print(f"DEBUG: Processing result with {len(result.boxes)} boxes")
                
                # Iterate through detected boxes
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = result.names[class_id]
                    
                    # Store detection
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'class_id': class_id
                    })
                    
                    # Track statistics by class
                    if class_name not in class_stats:
                        class_stats[class_name] = {
                            'count': 0,
                            'confidences': []
                        }
                    class_stats[class_name]['count'] += 1
                    class_stats[class_name]['confidences'].append(confidence)
                    
                    print(f"DEBUG: Detected {class_name} with confidence {confidence:.2f}")
            
            # Calculate average confidence per class
            detection_summary = []
            for class_name, stats in class_stats.items():
                avg_confidence = sum(stats['confidences']) / len(stats['confidences'])
                detection_summary.append({
                    'class': class_name,
                    'confidence': avg_confidence,
                    'count': stats['count']
                })
            
            # Sort by count (descending)
            detection_summary.sort(key=lambda x: x['count'], reverse=True)
            
            # Generate visualization using YOLO's built-in plot() method
            vis_path = self._plot_and_save(results[0], image_path)
            
            print(f"DEBUG: Visualization saved to {vis_path}")
            
            return {
                'detections': detection_summary,
                'detailed_detections': detections,
                'visualization': os.path.basename(vis_path) if vis_path else None,
                'success': True,
                'total_objects': len(detections),
                'total_classes': len(class_stats),
                'model_used': self.model_name,
                'source': 'yolo'
            }
            
        except Exception as e:
            import traceback
            print(f"❌ YOLO Error: {str(e)}")
            print(traceback.format_exc())
            return {
                'detections': [],
                'detailed_detections': [],
                'visualization': None,
                'success': False,
                'error': f'YOLO detection error: {str(e)}',
                'source': 'error'
            }
    
    def _plot_and_save(self, result, image_path: str) -> Optional[str]:
        """
        Use YOLO's built-in plot() method to draw bboxes and save
        
        Args:
            result: YOLO result object
            image_path: Path to original image
            
        Returns:
            Path to saved visualization or None if failed
        """
        try:
            print(f"DEBUG: Generating visualization using YOLO.plot()")
            
            # Get the plotted image from YOLO (numpy array in BGR)
            plot_image = result.plot()
            print(f"DEBUG: plot() returned image of shape {plot_image.shape}")
            
            # Convert BGR to RGB for PIL
            plot_rgb = cv2.cvtColor(plot_image, cv2.COLOR_BGR2RGB)
            print(f"DEBUG: Converted BGR to RGB")
            
            # Convert to PIL Image
            pil_image = Image.fromarray(plot_rgb)
            
            # Save visualization
            base, ext = os.path.splitext(image_path)
            vis_path = base + '_bbox' + ext
            
            print(f"DEBUG: Saving to {vis_path}")
            pil_image.save(vis_path, quality=95)
            print(f"DEBUG: ✅ Visualization saved successfully to {vis_path}")
            
            return vis_path
        
        except Exception as e:
            import traceback
            print(f"❌ ERROR in _plot_and_save: {str(e)}")
            print(traceback.format_exc())
            return None
    
    def _demo_detection(self, image_path: str) -> Dict:
        """Return mock detection result for demo mode, with visualization"""
        try:
            # Create simple demo visualization
            vis_path = self._create_demo_visualization(image_path)
        except Exception as e:
            print(f"DEBUG: Could not create demo visualization: {e}")
            vis_path = None
        
        demo_detections = [
            {"class": "Building", "confidence": 0.95, "count": 8},
            {"class": "Vehicle", "confidence": 0.87, "count": 3},
            {"class": "Tree", "confidence": 0.92, "count": 12},
            {"class": "Road", "confidence": 0.99, "count": 1},
            {"class": "Parking Lot", "confidence": 0.84, "count": 2}
        ]
        
        return {
            'detections': demo_detections,
            'detailed_detections': [],
            'visualization': os.path.basename(vis_path) if vis_path else None,
            'success': True,
            'total_objects': sum(d['count'] for d in demo_detections),
            'total_classes': len(demo_detections),
            'model_used': self.model_name,
            'source': 'demo'
        }
    
    def _create_demo_visualization(self, image_path: str) -> Optional[str]:
        """Create a simple demo visualization with sample boxes using OpenCV"""
        try:
            print(f"DEBUG: Creating demo visualization")
            
            # Read image with OpenCV (BGR format)
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            print(f"DEBUG: Image loaded, shape: {img_bgr.shape}")
            
            # Draw some demo boxes
            height, width = img_bgr.shape[:2]
            demo_boxes = [
                ((int(width*0.05), int(height*0.05)), (int(width*0.25), int(height*0.25)), "Building", 0.95),
                ((int(width*0.3), int(height*0.1)), (int(width*0.55), int(height*0.35)), "Vehicle", 0.87),
                ((int(width*0.2), int(height*0.4)), (int(width*0.45), int(height*0.65)), "Tree", 0.92),
            ]
            
            # BGR colors
            color_bgr = (0, 255, 0)  # Green
            
            for (x1, y1), (x2, y2), label, conf in demo_boxes:
                # Draw rectangle
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color_bgr, 2)
                
                # Draw label
                text = f"{label} {conf:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                
                # Background for text
                cv2.rectangle(img_bgr, (x1, y1 - text_size[1] - 5), 
                            (x1 + text_size[0] + 5, y1), color_bgr, -1)
                
                # Text
                cv2.putText(img_bgr, text, (x1, y1 - 5), font, font_scale, 
                          (255, 255, 255), thickness)
            
            # Save visualization (OpenCV saves in BGR)
            base, ext = os.path.splitext(image_path)
            vis_path = base + '_bbox' + ext
            
            cv2.imwrite(vis_path, img_bgr)
            print(f"DEBUG: ✅ Demo visualization saved to {vis_path}")
            
            return vis_path
        except Exception as e:
            import traceback
            print(f"ERROR in _create_demo_visualization: {e}")
            print(traceback.format_exc())
            return None


def get_image_info(image_path: str) -> Dict:
    """Get basic image metadata"""
    try:
        img = Image.open(image_path)
        return {
            'format': img.format,
            'size': img.size,
            'width': img.width,
            'height': img.height,
            'mode': img.mode,
            'success': True
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
