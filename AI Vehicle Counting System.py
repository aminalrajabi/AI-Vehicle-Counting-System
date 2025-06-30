import gradio as gr
from ultralytics import YOLO
import cv2
import os
import tempfile
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

# Solve OMP issue
# Set environment variable to allow duplicate OpenMP runtime initialization
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load YOLO model for vehicle detection
model = YOLO("yolov8n.pt")

# Allowed video file extensions for upload
ALLOWED_VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

# Global variables to temporarily store line coordinates for manual drawing
# These will be updated by user clicks and used by the "Draw Line" button
# (They will be re-defined as gr.State inside the Gradio interface, but kept here for clarity)
temp_x1 = None  # Temporary X1 coordinate for line start
temp_y1 = None  # Temporary Y1 coordinate for line start
temp_x2 = None  # Temporary X2 coordinate for line end
temp_y2 = None  # Temporary Y2 coordinate for line end
click_counter = 0  # Counter to track whether it's the first or second click

def cross_product(x1, y1, x2, y2):
    return x1 * y2 - y1 * x2

def validate_video_file(file_path):
    """Validate video file"""
    if not file_path or not os.path.exists(file_path):
        return False, "File not found"
    
    filename = os.path.basename(file_path).lower()
    suffix = os.path.splitext(filename)[1]
    
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        return False, f"Unsupported file format. Supported formats: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
    
    return True, "Valid"

def get_video_details(video_file):
    """Get video details and first frame"""
    if not video_file:
        # If no video, clear info and hide image
        return "No video uploaded", None, None, None, gr.update(value=None, interactive=False, visible=False)
    
    is_valid, error_msg = validate_video_file(file_path=video_file)
    if not is_valid:
        # If invalid, show error and hide image
        return error_msg, None, None, None, gr.update(value=None, interactive=False, visible=False)
    
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        # If cannot open, show error and hide image
        return "Could not open video", None, None, None, gr.update(value=None, interactive=False, visible=False)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    # Get first frame
    ret, first_frame_bgr = cap.read()
    cap.release()
    
    if not ret:
        # If cannot read frame, show error and hide image
        return "Could not read first frame", None, None, None, gr.update(value=None, interactive=False, visible=False)
    
    # Convert frame to RGB for Gradio display
    first_frame_rgb = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2RGB)
    
    info = f"""
    📹 **Video Details:**
    - Width: {width} pixels
    - Height: {height} pixels
    - Frame Rate: {fps:.2f} fps
    - Frame Count: {frame_count}
    - Duration: {duration:.2f} seconds
    """
    
    # Also return the first frame to the line_preview to allow clicking on it.
    # line_preview will be interactive to allow manual drawing.
    return info, first_frame_rgb, width, height, gr.update(value=first_frame_rgb, interactive=True, visible=True)

def draw_line_on_frame(video_file, x1, y1, x2, y2):
    """Draw line on first frame of video"""
    if not video_file:
        return "No video uploaded", None
    
    is_valid, error_msg = validate_video_file(file_path=video_file)
    if not is_valid:
        return error_msg, None
    
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return "Could not read video", None
    
    # Draw line
    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
    
    # Add start and end points
    cv2.circle(frame, (int(x1), int(y1)), 5, (255, 0, 0), -1)
    cv2.circle(frame, (int(x2), int(y2)), 5, (0, 0, 255), -1)
    
    # Add coordinate text
    cv2.putText(frame, f"Start: ({int(x1)}, {int(y1)})", (int(x1)+10, int(y1)-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"End: ({int(x2)}, {int(y2)})", (int(x2)+10, int(y2)-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Convert to RGB for Gradio display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return "Line drawn successfully! ✅", frame_rgb

def process_click_coordinates(video_file, current_x1, current_y1, current_x2, current_y2, current_click_counter, evt: gr.SelectData):
    """Process click coordinates for manual line drawing on line_preview"""
    if not video_file:
        return current_x1, current_y1, current_x2, current_y2, current_click_counter, "No video uploaded", None, gr.update(value=current_x1), gr.update(value=current_y1), gr.update(value=current_x2), gr.update(value=current_y2)

    x, y = evt.index[0], evt.index[1]

    if current_click_counter % 2 == 0: # First click (start point)
        new_x1, new_y1 = x, y
        new_x2, new_y2 = current_x2, current_y2 # Keep previous end if exists, or None
        info_msg = f"Start point set: ({x}, {y}). Click again for end point."
        new_click_counter = 1
    else: # Second click (end point)
        new_x1, new_y1 = current_x1, current_y1 # Keep previous start
        new_x2, new_y2 = x, y
        info_msg = f"End point set: ({x}, {y}). Line defined."
        new_click_counter = 2

    # Draw line dynamically on line_preview after each click
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return current_x1, current_y1, current_x2, current_y2, current_click_counter, "Could not read video for preview", None, gr.update(value=current_x1), gr.update(value=current_y1), gr.update(value=current_x2), gr.update(value=current_y2)

    # Use the current, possibly incomplete, line for drawing
    if new_x1 is not None and new_y1 is not None and new_x2 is not None and new_y2 is not None:
        cv2.line(frame, (int(new_x1), int(new_y1)), (int(new_x2), int(new_y2)), (0, 255, 0), 3)
        cv2.circle(frame, (int(new_x1), int(new_y1)), 5, (255, 0, 0), -1)
        cv2.circle(frame, (int(new_x2), int(new_y2)), 5, (0, 0, 255), -1)
    elif new_x1 is not None and new_y1 is not None: # Only start point clicked
        cv2.circle(frame, (int(new_x1), int(new_y1)), 5, (255, 0, 0), -1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return new_x1, new_y1, new_x2, new_y2, new_click_counter, info_msg, frame_rgb, gr.update(value=new_x1), gr.update(value=new_y1), gr.update(value=new_x2), gr.update(value=new_y2)

def count_vehicles(video_file, x1, y1, x2, y2, progress=gr.Progress()):
    """Count vehicles with progress display"""
    if not video_file:
        return "No video uploaded", None, None, None

    is_valid, error_msg = validate_video_file(video_file)
    if not is_valid:
        return error_msg, None, None, None

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        return "Could not open video", None, None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    line_start = (int(x1), int(y1))
    line_end = (int(x2), int(y2))

    count_up = 0
    count_down = 0
    vehicle_states = {}
    counted_ids = {}

    # Create a temporary directory for car images
    car_save_dir = tempfile.mkdtemp()

    # Clean the directory
    for f in os.listdir(car_save_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                os.remove(os.path.join(car_save_dir, f))
            except Exception:
                pass

    # Create output video in a temporary directory
    output_video_path = tempfile.mktemp(suffix=".mp4")

    # Use a more compatible codec
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # Changed from mp4v to H264
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    progress(0, desc="Starting video processing...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        progress_percent = frame_count / total_frames
        progress(progress_percent, desc=f"Processing frame {frame_count}/{total_frames}")

        results = model.track(frame, persist=True, verbose=False)

        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                track_id = int(box.id[0]) if box.id is not None else None

                if cls not in [2, 3, 5, 7] or track_id is None:
                    continue

                x1b, y1b, x2b, y2b = map(int, box.xyxy[0])
                cx = (x1b + x2b) // 2
                cy = (y1b + y2b) // 2

                dx_line = line_end[0] - line_start[0]
                dy_line = line_end[1] - line_start[1]
                dx_curr = cx - line_start[0]
                dy_curr = cy - line_start[1]
                curr_cross = cross_product(dx_line, dy_line, dx_curr, dy_curr)

                curr_state = "up" if curr_cross < 0 else "down"

                if track_id in vehicle_states:
                    prev_state = vehicle_states[track_id]
                    if prev_state != curr_state:
                        if curr_state == "down" and counted_ids.get(track_id) != "down":
                            count_down += 1
                            counted_ids[track_id] = "down"
                            car_crop = frame[y1b:y2b, x1b:x2b]
                            car_img_path = os.path.join(car_save_dir, f"car_{track_id}_{frame_count}.jpg")
                            cv2.imwrite(car_img_path, car_crop)
                        elif curr_state == "up" and counted_ids.get(track_id) != "up":
                            count_up += 1
                            counted_ids[track_id] = "up"
                            car_crop = frame[y1b:y2b, x1b:x2b]
                            car_img_path = os.path.join(car_save_dir, f"car_{track_id}_{frame_count}.jpg")
                            cv2.imwrite(car_img_path, car_crop)

                vehicle_states[track_id] = curr_state

                color = (0, 255, 0) if curr_state == "up" else (0, 0, 255)
                cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), color, 2)
                cv2.putText(frame, f"ID:{track_id} {curr_state}", 
                            (x1b, y1b - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.line(frame, line_start, line_end, (0, 255, 0), 3)
        cv2.circle(frame, line_start, 5, (255, 0, 0), -1)
        cv2.circle(frame, line_end, 5, (0, 0, 255), -1)

        cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"Up: {count_up}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Down: {count_down}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release()

    # Short wait to ensure writing is complete
    import time
    time.sleep(0.5)

    # Check if the file exists and its size
    if not os.path.exists(output_video_path) or os.path.getsize(output_video_path) == 0:
        return "Error: Failed to create output video", None, None, None

    # Analyze car images
    car_analysis_img = analyze_folder_images(car_save_dir)

    # Draw statistics chart
    results_chart = create_results_chart(count_up, count_down)

    

    # Clean up temporary image directory
    try:
        shutil.rmtree(car_save_dir)
    except:
        pass

    return results_chart, output_video_path, car_analysis_img
def create_results_chart(count_up, count_down):
    """Create enhanced results chart"""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1e1e2e')
    ax.set_facecolor('#1e1e2e')
    
    categories = ['Going Up', 'Going Down']
    values = [count_up, count_down]
    
    # Create gradient colors
    colors = ['#00ff88', '#ff4757']
    
    # Create bars with enhanced styling
    bars = ax.bar(categories, values, color=colors, alpha=0.8, width=0.6)
    
    # Add gradient effect
    for bar, color in zip(bars, colors):
        bar.set_edgecolor('white')
        bar.set_linewidth(2)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                f'{value}', ha='center', va='bottom', fontsize=16, 
                fontweight='bold', color='white')
    
    # Styling
    ax.set_ylabel('Number of Vehicles', fontsize=14, color='white', fontweight='bold')
    ax.set_title('Vehicle Counting Statistics', fontsize=18, fontweight='bold', 
                color='white', pad=20)
    ax.grid(axis='y', alpha=0.3, color='gray')
    ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 10)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    
    # Color the tick labels
    ax.tick_params(colors='white', labelsize=12)
    
    # Add total count
    total = count_up + count_down
    ax.text(0.5, 0.95, f'Total: {total} vehicles', transform=ax.transAxes, 
            ha='center', va='top', fontsize=14, fontweight='bold', 
            color='#ffd700', bbox=dict(boxstyle='round,pad=0.3', facecolor='#333', alpha=0.8))
    
    plt.tight_layout()
    
    # Save chart
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                facecolor='#1e1e2e', edgecolor='none')
    buf.seek(0)
    
    plt.close()
    
    # Convert to PIL image
    chart_image = Image.open(buf)
    
    return chart_image

# Analyze captured car images
import base64, requests, json, pandas as pd, matplotlib.pyplot as plt, io
from PIL import Image

def analyze_folder_images(image_folder_path):
    api_key = ""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    results = []
    image_files = [f for f in os.listdir(image_folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for image_file in image_files:
        with open(os.path.join(image_folder_path, image_file), "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                            "You are a vehicle recognition system. Analyze the vehicle in the image and provide your best possible guess of the car's **specific model** (not just the brand), along with the color. Focus on visible features such as shape, headlights, taillights, grille, and overall body design.\n\n"
                            "Always try to identify the specific model name (e.g., 'Civic', 'Corolla', 'Tucson'). Avoid general terms or brand names alone. If the model is unclear, make your best educated guess based on appearance — do not use 'Unknown' unless the car is fully obscured.\n\n"
                            "Respond only with this JSON:\n"
                            "{\n"
                            "  \"Color\": \"Car color, or 'None' if not visible\",\n"
                            "  \"Model\": \"Most likely specific model of the car based on visual features\"\n"
                            "}\n\n"
                            "Return JSON only. No extra explanation or text."
)
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0
        }
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            reply = response.json()["choices"][0]["message"]["content"]
            try:
                if reply.strip().startswith("```json"):
                    reply = reply.strip()[7:]
                elif reply.strip().startswith("```"):
                    reply = reply.strip()[3:]
                if reply.strip().endswith("```"):
                    reply = reply.strip()[:-3]
                result = json.loads(reply)
                
                # Process the model: split and take only the first word
                if result['Model'] and isinstance(result['Model'], str):
                    result['Model'] = result['Model'].split()[0]
                results.append(result)
            except Exception:
                results.append({"Color": "Unknown", "Model": "Unknown", "Image": image_file})
        else:
            results.append({"Color": "Error", "Model": "Error", "Image": image_file})
    # Draw charts
    df = pd.DataFrame(results)
    if not df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        color_counts = df['Color'].value_counts()
        color_counts.plot(kind='bar', color='skyblue', ax=axes[0])
        axes[0].set_title('Car Color Distribution')
        axes[0].set_xlabel('Color')
        axes[0].set_ylabel('Number of Cars')
        model_counts = df['Model'].value_counts()
        model_counts.plot(kind='bar', color='orange', ax=axes[1])
        axes[1].set_title('Car Model Distribution')
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('Number of Cars')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        return img
    else:
        return None

def show_main_interface():
    """Show main interface and hide instructions"""
    return gr.update(visible=False), gr.update(visible=True)

# Create Gradio interface
def create_gradio_interface():
    with gr.Blocks(title="AI Vehicle Counting System", theme=gr.themes.Soft()) as demo:
        # ================== Page 1: Video Upload ==================
        upload_page = gr.Column(visible=True)
        with upload_page:
            # Box 1: Main title (top of the page)
            gr.Markdown(
                """
                <div style='text-align:center; margin-bottom:20px;'>
                    <h1 style='color:#007bff;'>🚗 AI Vehicle Counting System</h1>
                    <h3 style='color:#333;'>Upload a video to start vehicle counting</h3>
                </div>
                """
            )
            with gr.Row():
                with gr.Column(scale=1, min_width=400):
                    # Box 2: Instructions box (middle of the first page)
                    gr.Markdown(
                        """
                        <div style='background:#f5f7fa; border-radius:10px; padding:30px; box-shadow:0 2px 8px #0001; margin-bottom:10px;'>
                            <h4 style='color:#007bff; margin-bottom:10px;'>Step 1: Upload Video</h4>
                            <p style='color:#555;'>Please upload a video file (MP4, AVI, MOV, MKV, WEBM) to continue.</p>
                        </div>
                        """
                    )
                    # Box 3: Video upload (below the instructions box)
                    video_input = gr.File(
                        label="Upload Video", 
                        file_types=[".mp4", ".avi", ".mov", ".mkv", ".webm"]
                    )
        # ================== Page 2: Video Analysis ==================
        steps_page = gr.Column(visible=False)
        with steps_page:
            # Box 4: Main title for the second page (top of the page)
            gr.Markdown(
                """
                <div style='text-align:center; margin-bottom:20px;'>
                    <h1 style='color:#007bff;'>🚗 AI Vehicle Counting System</h1>
                    <h3 style='color:#333;'>Set the counting line and start analysis</h3>
                </div>
                """
            )
            # Box 5: CSS for images (hidden - controls styling)
            gr.HTML(
                """
                <style>
                #line-preview-fullwidth img, #line-preview-fullwidth canvas {
                    width: 100% !important;
                    max-width: 100vw !important;
                    min-height: 400px;
                    border-radius: 12px;
                    box-shadow: 0 2px 12px #0002;
                    margin-bottom: 10px;
                }
                </style>
                """
            )
            # Row 1: Video information and first frame
            with gr.Row():
                with gr.Column(scale=1):
                    # Box 6: Video information (hidden initially)
                    video_info = gr.Markdown("", visible=False)
                    # Box 7: First frame (hidden initially)
                    first_frame = gr.Image(
                        label="First Frame", 
                        interactive=False, 
                        visible=False
                    )
            # Box 8: Line preview (full width - for clicking and setting the line)
            line_preview = gr.Image(
                label="Line Preview (Click to draw line)", 
                interactive=True, 
                elem_id="line-preview-fullwidth"
            )
            # Row 2: Line settings
            with gr.Row():
                with gr.Column(scale=2):
                    # Box 9: Line settings title
                    gr.Markdown("### 📍 Set Counting Line")
                    # Boxes 10 and 11: Tabs for manual drawing and coordinates
                    with gr.Tabs():
                        with gr.Tab("Manual Drawing"):
                            gr.Markdown("Click on the preview image to set line points.")
                            # Box 12: Click information
                            click_info = gr.Textbox(
                                label="Click Info", 
                                interactive=False
                            )
                        with gr.Tab("Enter Coordinates"):
                            # Sub-row 1: Start point
                            with gr.Row():
                                # Box 13: X1
                                x1_input = gr.Number(
                                    label="X1 (Line Start)", 
                                    value=100
                                )
                                # Box 14: Y1
                                y1_input = gr.Number(
                                    label="Y1 (Line Start)", 
                                    value=300
                                )
                            # Sub-row 2: End point
                            with gr.Row():
                                # Box 15: X2
                                x2_input = gr.Number(
                                    label="X2 (Line End)", 
                                    value=500
                                )
                                # Box 16: Y2
                                y2_input = gr.Number(
                                    label="Y2 (Line End)", 
                                    value=300
                                )
            # Row 3: Start button
            with gr.Row():
                # Box 17: Start counting button
                start_btn = gr.Button(
                    "🚦 Start Counting", 
                    variant="primary", 
                    size="lg"
                )
            # Row 4: Results
            with gr.Row():
                # Box 18: Results text
                result_text = gr.Markdown("📊 Results will appear here")
            # Row 5: Processed video and charts
            with gr.Row():
                # Box 19: Processed video (one-third width)
                processed_video = gr.Video(label="Processed Video", format="mp4", show_download_button=True)
                # Box 20: Statistics chart (one-third width)
                results_chart = gr.Image(
                    label="Statistics Chart", 
                    scale=1
                )
                # Box 21: Car analysis chart (one-third width)
                car_analysis_chart = gr.Image(
                    label="Car Color/Model Analysis", 
                    scale=1
                )
        # ================== State Variables (Hidden) ==================
        state_x1 = gr.State(None)           # X1 state
        state_y1 = gr.State(None)           # Y1 state
        state_x2 = gr.State(None)           # X2 state
        state_y2 = gr.State(None)           # Y2 state
        state_click_counter = gr.State(0)   # Click counter
        state_video_file = gr.State(None)   # Video file

        # Page navigation logic
        def go_to_steps(video_file):
            # Reset state variables on each new video
            video_info, first_frame_rgb, width, height, line_preview_update = get_video_details(video_file)
            return (
                gr.update(visible=False),  # Hide video upload page
                gr.update(visible=True),  # Show steps page
                video_info,  # Video information
                first_frame_rgb,  # First frame
                video_file,  # Store video in state
                None, None, None, None, 0,  # x1, y1, x2, y2, click_counter
                line_preview_update  # Update first frame display in line_preview
            )

        video_input.change(
            fn=go_to_steps,
            inputs=[video_input],
            outputs=[upload_page, steps_page, video_info, first_frame, state_video_file, state_x1, state_y1, state_x2, state_y2, state_click_counter, line_preview]
        )
        line_preview.select(
            fn=process_click_coordinates,
            inputs=[state_video_file, state_x1, state_y1, state_x2, state_y2, state_click_counter],
            outputs=[state_x1, state_y1, state_x2, state_y2, state_click_counter, click_info, line_preview, x1_input, y1_input, x2_input, y2_input]
        )
        start_btn.click(
            fn=count_vehicles,
            inputs=[state_video_file, x1_input, y1_input, x2_input, y2_input],
            outputs=[result_text, results_chart, processed_video, car_analysis_chart]
        )
        return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(server_port=7862, share=False, show_error=True)
