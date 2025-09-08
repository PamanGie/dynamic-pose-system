# Dynamic Custom Keypoint Pose Annotation & Inference System

A comprehensive system for creating custom pose datasets and performing real-time pose detection with dynamic keypoint configurations and custom skeleton visualization.

## 🎥 Demo Videos

### Pose Annotator Demo
[![Annotator Demo](https://img.youtube.com/vi/59HM-KuoeJw/maxresdefault.jpg)]([https://youtu.be/59HM-KuoeJw](https://youtu.be/59HM-KuoeJw))

### Custom Inference Demo  
[![Inference Demo](https://img.youtube.com/vi/k80OHvJaKlk/maxresdefault.jpg)](https://youtu.be/k80OHvJaKlk)

## 🌟 Features

### 📝 Dynamic Pose Annotator
- **Custom Keypoint Definition** - Define any keypoints for your specific use case
- **Sequential Connection System** - Automatic keypoint chaining (A→B→C→D...)
- **Auto-copy Frame Feature** - Efficient annotation with automatic keypoint copying to next frame
- **Drag & Drop Editing** - Real-time keypoint position adjustment
- **Insert on Line** - Add keypoints between existing connections
- **Multiple Input Formats** - Support for videos (.mp4, .avi, .mov) and image sequences

### 🎯 Dynamic Custom Skeleton Inference
- **Flexible Skeleton Visualization** - Custom bone connections based on your keypoints
- **Real-time Detection** - Live pose detection with custom skeleton overlay
- **Color-coded Connections** - Different colors for different body parts/connections
- **Smart Classification** - Configurable pose classification logic
- **Multiple Output Formats** - Video, image, and real-time webcam support

### 🚀 YOLO11 Integration
- **State-of-the-art Accuracy** - Built on YOLO11 pose estimation
- **Custom Training Pipeline** - Easy training with your annotated dataset
- **High Performance** - Optimized for real-time applications
- **GPU Acceleration** - CUDA support for fast inference

## 📋 Requirements

### Minimum Requirements
- **CPU**: Intel i5 / AMD Ryzen 5
- **RAM**: 8GB
- **GPU**: GTX 1060 6GB (optional, but recommended)
- **Storage**: 10GB free space
- **Python**: 3.7+

### Recommended Requirements
- **CPU**: Intel i7 / AMD Ryzen 7
- **RAM**: 16GB
- **GPU**: RTX 3060 12GB or higher
- **Storage**: SSD with 20GB free space
- **CUDA**: 11.8+ for GPU acceleration

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/dynamic-pose-keypoint-annotator.git
cd dynamic-pose-keypoint-annotator
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install YOLO11 (for training and inference)
```bash
pip install ultralytics
```

## 🎮 Usage

### Dynamic Pose Annotator

#### Starting the Annotator
```bash
python pose_annotator.py
```

#### Workflow
1. **Load Media**: Click "Load Video" or "Load Image Folder"
2. **Annotate Keypoints**: Click on image to add custom keypoints
3. **Define Names**: Enter descriptive names for each keypoint (e.g., "head", "hand", "tool")
4. **Sequential Connections**: Keypoints automatically connect in sequence
5. **Navigate Frames**: Use arrow keys or navigation buttons
6. **Auto-copy**: Keypoints automatically copy to next frame for efficiency
7. **Export Dataset**: Export in YOLO format for training

#### Key Features
- **Right-click**: Delete keypoint
- **Drag**: Move keypoint position
- **Click on line**: Insert keypoint between connections
- **Break Chain**: Start new connection sequence
- **Zoom**: Mouse wheel or buttons for detailed annotation

### Training Custom Model

#### Prepare Dataset
After annotation, your dataset structure should be:
```
dataset/
├── images/           # Training images
├── labels/           # YOLO pose labels
└── dataset.yaml      # Configuration file
```

#### Train Model
```bash
# Basic training
yolo pose train data=dataset.yaml model=yolo11m-pose.pt epochs=300

# Advanced training with custom parameters
yolo pose train data=dataset.yaml model=yolo11m-pose.pt epochs=300 imgsz=640 batch=8 device=0
```

#### Model Selection
| Model | Size | GPU Memory | Speed | Accuracy |
|-------|------|------------|-------|----------|
| yolo11n-pose | ~6MB | ~2GB | Fastest | Good |
| yolo11s-pose | ~22MB | ~4GB | Fast | Better |
| yolo11m-pose | ~52MB | ~8GB | Medium | Very Good |
| yolo11l-pose | ~100MB | ~12GB | Slow | Excellent |

### Dynamic Custom Skeleton Inference

#### Basic Inference
```bash
# Video inference
yolo pose predict model=best.pt source=video.mp4 save=True

# Image inference
yolo pose predict model=best.pt source=image.jpg save=True

# Real-time webcam
yolo pose predict model=best.pt source=0 show=True
```

#### Custom Skeleton Visualization
```bash
python custom_inference.py
```

**Menu Options:**
1. Process video with custom skeleton
2. Process single image
3. Quick test mode

The custom inference script provides:
- **Dynamic skeleton connections** based on your keypoint definitions
- **Color-coded body parts** for clear visualization
- **Real-time classification** based on custom logic
- **Performance statistics** and frame-by-frame analysis

## 📊 Performance

### Training Results (Example)
- **Training Time**: ~1.6 hours (300 epochs)
- **Accuracy**: 99.5% mAP50
- **Model Size**: 42.3MB
- **Parameters**: 20M+

### Inference Performance
| Hardware | FPS | Resolution |
|----------|-----|------------|
| RTX 3060 | 45-60 | 640x640 |
| RTX 4080 | 80-100 | 640x640 |
| RTX 4090 | 120+ | 640x640 |
| CPU i7 | 8-12 | 640x640 |

## 🎯 Applications

This system can be adapted for various pose detection tasks:

- **Sports Analysis** - Custom keypoints for specific sports movements
- **Medical Applications** - Patient movement analysis and rehabilitation
- **Industrial Safety** - Worker pose monitoring and safety compliance
- **Security & Surveillance** - Custom behavior detection
- **Entertainment** - Motion capture and animation
- **Research** - Custom pose estimation for specific domains

## 📁 Project Structure

```
dynamic-pose-system/
├── pose_annotator.py              # Main annotation tool
├── custom_inference.py            # Custom skeleton inference
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── examples/                      # Example datasets and configs
│   ├── sample_dataset/
│   └── sample_configs/
├── docs/                          # Documentation
└── utils/                         # Utility scripts
    ├── data_converter.py
    └── visualization.py
```

## 🔧 Configuration

### Dataset Configuration (dataset.yaml)
```yaml
# Custom pose detection dataset
path: /path/to/dataset
train: images
val: images

# Keypoints configuration
kpt_shape: [N, 3]  # N = number of your custom keypoints
flip_idx: [0, 1, 2, ...]  # Keypoint flip indices

# Classes
nc: 1
names: ['person']

# Custom keypoint names
keypoint_names: ['point1', 'point2', 'point3', ...]

# Custom skeleton connections
skeleton: [
  [0, 1], [1, 2], [2, 3], ...  # Define your connections
]
```

### Custom Inference Configuration
The inference script allows you to define:
- **Custom skeleton connections** between keypoints
- **Color coding** for different body parts
- **Classification logic** based on keypoint positions
- **Visualization styles** and thickness

## 🚀 Getting Started

1. **Clone the repository**
2. **Install dependencies**
3. **Run the annotator**: `python pose_annotator.py`
4. **Load your media** (video or images)
5. **Define custom keypoints** for your use case
6. **Annotate frames** with sequential connections
7. **Export dataset** in YOLO format
8. **Train model** with your custom keypoints
9. **Run inference** with custom skeleton visualization

# Custom Inference Setup Tutorial

## Overview
This guide explains how to customize the inference script (`custom_inference.py`) for your specific pose detection needs.

## Configuration Steps

### 1. Define Your Keypoint Names
Replace the `KEYPOINT_NAMES` array with your actual keypoint names from annotation:

```python
KEYPOINT_NAMES = [
    'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'special_object', 'left_hip', 'right_hip',
    # Add all your keypoints here...
]
```

### 2. Configure Skeleton Connections
Define how your keypoints connect to each other in `CUSTOM_SKELETON`:

```python
CUSTOM_SKELETON = [
    (0, 1),   # head -> left_shoulder
    (0, 2),   # head -> right_shoulder  
    (1, 3),   # left_shoulder -> left_elbow
    (3, 5),   # left_elbow -> left_wrist
    (5, 7),   # left_wrist -> special_object (KEY CONNECTION)
    # Add more connections...
]
```

### 3. Set Special Keypoint Index
Identify which keypoint is most important for your behavior detection:

```python
SPECIAL_KEYPOINT_INDEX = 7  # Index of your key behavior keypoint
```

### 4. Customize Detection Logic
Modify the `is_behavior_detected()` function for your specific use case:

```python
def is_behavior_detected(keypoints):
    """Customize this function for your behavior detection"""
    if len(keypoints) > SPECIAL_KEYPOINT_INDEX:
        special_confidence = keypoints[SPECIAL_KEYPOINT_INDEX][2]
        return special_confidence > 0.5
    return False
```

### 5. Update Labels and Thresholds
Set appropriate labels and confidence thresholds:

```python
POSITIVE_LABEL = "BEHAVIOR_DETECTED"     # Your behavior name
NEGATIVE_LABEL = "NO_BEHAVIOR"           
POSITIVE_CLASS_NAME = "person_with_tool" # Example: tool usage
NEGATIVE_CLASS_NAME = "person"
BEHAVIOR_THRESHOLD = 0.5                 # Adjust as needed
```

### 6. Customize Colors (Optional)
Modify connection colors for different body parts:

```python
CONNECTION_COLORS = {
    'head': (0, 255, 255),        # Yellow
    'arms': (0, 255, 0),          # Green  
    'special': (0, 0, 255),       # Red - for key behavior
    'body': (255, 0, 0),          # Blue
    'legs': (255, 255, 0)         # Cyan
}
```

## Usage Examples

### Example 1: Smoking Detection
```python
KEYPOINT_NAMES = ['head', 'left_shoulder', 'right_shoulder', 'left_wrist', 'right_wrist', 'cigarette']
SPECIAL_KEYPOINT_INDEX = 5  # cigarette
POSITIVE_LABEL = "SMOKING_DETECTED"
POSITIVE_CLASS_NAME = "smoker"
```

### Example 2: Tool Usage Detection
```python
KEYPOINT_NAMES = ['head', 'left_shoulder', 'right_shoulder', 'left_hand', 'right_hand', 'tool']
SPECIAL_KEYPOINT_INDEX = 5  # tool
POSITIVE_LABEL = "TOOL_USAGE_DETECTED"  
POSITIVE_CLASS_NAME = "worker"
```

### Example 3: Sports Movement Analysis
```python
KEYPOINT_NAMES = ['head', 'left_shoulder', 'right_shoulder', 'left_hand', 'right_hand', 'ball']
SPECIAL_KEYPOINT_INDEX = 5  # ball
POSITIVE_LABEL = "BALL_CONTACT"
POSITIVE_CLASS_NAME = "player_with_ball"
```

## Running Custom Inference

### 1. Basic Usage
```bash
python custom_inference.py
```

### 2. Menu Options
- **Option 1**: Process video with custom skeleton
- **Option 2**: Process single image
- **Option 3**: Quick test mode

### 3. Expected Output
- Bounding boxes with custom labels
- Colored skeleton connections
- Real-time behavior detection status
- Frame-by-frame statistics

## Advanced Customization

### Multi-Keypoint Behavior Detection
For complex behaviors involving multiple keypoints:

```python
def is_behavior_detected(keypoints):
    """Advanced multi-keypoint detection"""
    # Check multiple conditions
    has_object = len(keypoints) > OBJECT_INDEX and keypoints[OBJECT_INDEX][2] > 0.5
    hand_position = len(keypoints) > HAND_INDEX and keypoints[HAND_INDEX][2] > 0.5
    
    if has_object and hand_position:
        # Calculate distance between hand and object
        hand_pos = keypoints[HAND_INDEX][:2]
        object_pos = keypoints[OBJECT_INDEX][:2]
        distance = np.linalg.norm(np.array(hand_pos) - np.array(object_pos))
        return distance < 50  # proximity threshold
    
    return False
```

### Distance-Based Analysis
For pose analysis based on keypoint distances:

```python
def analyze_pose_distance(keypoints):
    """Analyze behavior based on keypoint distances"""
    if len(keypoints) > max(HAND_INDEX, MOUTH_INDEX):
        hand_pos = keypoints[HAND_INDEX][:2]
        mouth_pos = keypoints[MOUTH_INDEX][:2]
        distance = np.linalg.norm(np.array(hand_pos) - np.array(mouth_pos))
        
        # Return behavior probability based on distance
        return distance < 80  # Adjust threshold as needed
```

## Troubleshooting

### Common Issues

1. **Keypoint Index Errors**
   - Ensure `SPECIAL_KEYPOINT_INDEX` is within range
   - Check that keypoint names match your annotation

2. **No Skeleton Visible**
   - Verify skeleton connections use correct indices
   - Check if keypoints have sufficient confidence

3. **Wrong Behavior Detection**
   - Adjust `BEHAVIOR_THRESHOLD` value
   - Modify detection logic in `is_behavior_detected()`

4. **Poor Performance**
   - Reduce confidence threshold for faster detection
   - Skip frames for real-time processing

### Debugging Tips

1. **Print keypoint information**:
```python
print(f"Detected keypoints: {len(keypoints)}")
print(f"Special keypoint confidence: {keypoints[SPECIAL_KEYPOINT_INDEX][2]}")
```

2. **Visualize keypoint indices**:
```python
# Add index numbers to keypoint labels
label = f"{KEYPOINT_NAMES[i]} ({i})"
```

3. **Test with single image first** before processing videos

## Performance Optimization

### For Real-time Applications
```python
# Skip frames for better performance
frame_skip = 2  # Process every 2nd frame
if frame_count % frame_skip == 0:
    # Process frame
```

### For High Accuracy
```python
# Use higher confidence thresholds
CONFIDENCE_THRESHOLD = 0.7
BEHAVIOR_THRESHOLD = 0.7
```

## Integration with Other Systems

### Save Detection Results
```python
# Save results to JSON
results = {
    'frame': frame_count,
    'behavior_detected': behavior_detected,
    'confidence': confidence_score,
    'timestamp': time.time()
}

with open('detection_log.json', 'a') as f:
    json.dump(results, f)
```

### Real-time Alerts
```python
if behavior_detected:
    # Send alert, log event, trigger action, etc.
    print("ALERT: Behavior detected!")
    # webhook_notify(behavior_data)
```

This tutorial should help you customize the inference script for any pose-based behavior detection task.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Contributing Guidelines
- Follow PEP 8 style guidelines
- Add documentation for new features
- Include examples for new functionality
- Test on different platforms

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics** - YOLO11 framework
- **OpenCV** - Computer vision library
- **PyTorch** - Deep learning backend
- **Tkinter** - GUI framework

---

**Brought You By from HPC Lab of Tunghai University - Made with ❤️ for the computer vision community**

*Star ⭐ this repo if you find it useful!*
