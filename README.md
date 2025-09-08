# Dynamic Custom Keypoint Pose Annotation & Inference System

A comprehensive system for creating custom pose datasets and performing real-time pose detection with dynamic keypoint configurations and custom skeleton visualization.

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
