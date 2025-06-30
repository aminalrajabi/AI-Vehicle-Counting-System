# 🚗 AI Vehicle Counting System

## 📖 Overview

The AI Vehicle Counting System is an advanced computer vision application that automatically detects and counts vehicles crossing a user-defined line in video footage. Built using YOLOv8 for precise object detection and tracking, this intelligent system not only provides real-time vehicle counting but also leverages the power of OpenAI's GPT-4O vision model to analyze captured vehicle images, identifying specific vehicle colors and models with remarkable accuracy. This dual-AI approach combines state-of-the-art object detection with advanced visual intelligence to deliver comprehensive traffic analytics.

## ✨ Features

- **Real-time Vehicle Detection**: Uses YOLOv8 for accurate vehicle detection
- **Bidirectional Counting**: Counts vehicles moving in both directions (up/down)
- **Interactive Line Drawing**: Click-based interface for setting counting lines
- **Vehicle Analytics**: AI-powered color and model identification
- **Visual Results**: Comprehensive charts and processed video output
- **Multi-format Support**: Supports MP4, AVI, MOV, MKV, WEBM video formats

## 🛠️ Installation

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install gradio
pip install ultralytics
pip install opencv-python
pip install numpy
pip install Pillow
pip install matplotlib
pip install requests
pip install pandas
```

### Required Models

The system automatically downloads the YOLOv8 nano model (`yolov8n.pt`) on first run.

## 🚀 Usage

### Starting the Application

```bash
python AI Vehicle Counting System.py
```

The application will launch on `http://localhost:7862`

### Step-by-Step Guide

1. **Upload Video**: Select a video file from your device
2. **Set Counting Line**: Click two points on the video frame to define the counting line
3. **Start Analysis**: Click "Start Counting" to begin processing
4. **View Results**: Review the counting statistics and analytics

## 🔧 System Architecture

### Core Components

#### 1. Video Processing Module
```python
def validate_video_file(file_path)
def get_video_details(video_file)
```
- Validates input video formats
- Extracts video metadata (resolution, fps, duration)
- Retrieves first frame for line drawing

#### 2. Line Drawing Interface
```python
def process_click_coordinates(...)
def draw_line_on_frame(...)
```
- Interactive click-based line definition
- Real-time visual feedback
- Coordinate validation and storage

#### 3. Vehicle Detection & Tracking
```python
def count_vehicles(video_file, x1, y1, x2, y2, progress)
```
- YOLOv8-based object detection
- Multi-object tracking with unique IDs
- Cross-product algorithm for direction detection

#### 4. Analytics Engine
```python
def analyze_folder_images(image_folder_path)
def create_results_chart(count_up, count_down)
```
- AI-powered vehicle identification using GPT-4O
- Statistical visualization
- Color and model distribution analysis

## 📊 Technical Details

### Vehicle Detection Classes

The system detects the following vehicle types (COCO dataset classes):
- **Class 2**: Car
- **Class 3**: Motorcycle
- **Class 5**: Bus
- **Class 7**: Truck

### Direction Detection Algorithm

Uses cross-product calculation to determine vehicle direction:

```python
def cross_product(x1, y1, x2, y2):
    return x1 * y2 - y1 * x2
```

- **Positive cross-product**: Vehicle moving "down"
- **Negative cross-product**: Vehicle moving "up"

### Counting Logic

1. **State Tracking**: Each vehicle maintains a state (up/down)
2. **Transition Detection**: Counts when vehicle crosses line (state change)
3. **Duplicate Prevention**: Each vehicle ID counted only once per direction

### AI-Powered Analytics

Integrates with OpenAI's GPT-4O API for:
- Vehicle color identification
- Specific model recognition
- Confidence-based classification

## 🎯 Key Functions Breakdown

### Video Upload & Validation
```python
def validate_video_file(file_path):
```
- Checks file existence and format
- Validates against supported extensions
- Returns validation status and error messages

### Interactive Line Drawing
```python
def process_click_coordinates(...):
```
- Handles mouse click events on video frame
- Manages two-point line definition
- Updates UI with real-time feedback
- Stores coordinates in state variables

### Main Counting Engine
```python
def count_vehicles(video_file, x1, y1, x2, y2, progress):
```
**Process Flow:**
1. **Initialization**: Set up video capture and output writer
2. **Frame Processing**: Process each frame with YOLOv8
3. **Object Tracking**: Maintain vehicle IDs across frames
4. **Direction Calculation**: Use cross-product for direction detection
5. **State Management**: Track vehicle states and transitions
6. **Image Capture**: Save vehicle crops for analysis
7. **Video Output**: Generate annotated output video

### Vehicle Analytics
```python
def analyze_folder_images(image_folder_path):
```
- Processes captured vehicle images
- Sends images to GPT-4O API
- Extracts color and model information
- Generates distribution charts

### Results Visualization
```python
def create_results_chart(count_up, count_down):
```
- Creates professional statistical charts
- Implements dark theme styling
- Displays bidirectional counts
- Shows total vehicle count

## 🖥️ User Interface

### Page Structure

#### Page 1: Video Upload
- **Title Section**: Application branding
- **Instructions**: Step-by-step guidance
- **File Upload**: Drag-and-drop video input

#### Page 2: Analysis Interface
- **Video Information**: Displays video metadata
- **Line Drawing Canvas**: Interactive frame for line definition
- **Coordinate Input**: Manual coordinate entry option
- **Control Panel**: Start counting button
- **Results Display**: Statistics and visualizations

### State Management

The application uses Gradio's state system:
- `state_x1, state_y1`: Line start coordinates
- `state_x2, state_y2`: Line end coordinates
- `state_click_counter`: Click sequence tracking
- `state_video_file`: Current video file reference

## 🔄 Processing Pipeline

### 1. Initialization Phase
```
Video Upload → Validation → Metadata Extraction → First Frame Display
```

### 2. Configuration Phase
```
Line Drawing → Coordinate Storage → Visual Preview → Validation
```

### 3. Processing Phase
```
Video Loading → Frame-by-Frame Analysis → Object Detection → 
Tracking → Direction Calculation → Counting → Image Capture
```

### 4. Analysis Phase
```
Image Processing → AI Analysis → Chart Generation → Results Display
```

### 5. Output Generation
```
Annotated Video → Statistics Chart → Analytics Chart → Summary Report
```



## 🔧 Configuration

### Environment Variables
```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Resolves OpenMP conflicts
```

### API Configuration
Update the OpenAI API key in the `analyze_folder_images` function:
```python
api_key = "your-openai-api-key-here"
```

### Video Codec Settings
```python
fourcc = cv2.VideoWriter_fourcc(*'H264')  # Video compression
```

## 📊 Output Format

### Visual Outputs
1. **Statistics Chart**: Bar chart showing directional counts
2. **Analytics Chart**: Color and model distribution
3. **Processed Video**: Annotated video with tracking visualization

---