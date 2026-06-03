import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# --- PAGE CONFIG ---
st.set_page_config(page_title="HowlingWolfs Detection", page_icon="🐺", layout="wide")
st.title("🐺 HowlingWolfs: Vehicle Detection & Tracking")


# --- MODEL LOADING ---
@st.cache_resource
def load_model(model_path="best.pt"):
    try:
        # Fallback to yolov8n.pt if best.pt is not found for testing purposes
        if not os.path.exists(model_path):
            st.warning(f"'{model_path}' not found. Falling back to 'yolov8n.pt'")
            model_path = "yolov8n.pt"
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Load the model
model = load_model("best.pt")

# --- SIDEBAR CONFIG ---
st.sidebar.header("Configuration")
media_type = st.sidebar.radio("Select Media Type", ["Image", "Video"])

# Vehicle classes from COCO (2=car, 3=motorbike, 5=bus, 7=truck)
VEHICLE_CLASSES = [2, 3, 5, 7]

# --- IMAGE PROCESSING ---
if media_type == "Image":
    st.header("Image Detection")
    uploaded_image = st.file_uploader("Upload an Image", type=['png', 'jpg', 'jpeg'])

    if uploaded_image is not None:
        # Convert the uploaded file to an opencv image
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Display original image
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)

        if st.button("Detect Objects"):
            with st.spinner("Processing..."):
                results = model(image)
                annotated_img = results[0].plot()

                # Display output
                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Detected Image",
                         use_container_width=True)

# --- VIDEO PROCESSING ---
elif media_type == "Video":
    st.header("Video Tracking & Counting")
    uploaded_video = st.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi'])

    if uploaded_video is not None:
        if st.button("Process Video"):
            # Save uploaded video to a temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            video_path = tfile.name

            cap = cv2.VideoCapture(video_path)

            # Setup video writer to a temporary output file
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            # Note: browsers prefer H264 (libx264). We use mp4v for processing and provide a download.
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_file.name, fourcc, fps, (frame_width, frame_height))

            counted_ids = set()

            # UI Elements for progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            frame_window = st.empty()  # Used to show the video processing in real-time

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                resized = cv2.resize(frame, (640, 640))
                results = model.track(resized, persist=True, verbose=False)
                boxes = results[0].boxes

                if boxes is not None and boxes.id is not None:
                    for box, cls, track_id in zip(boxes.xyxy, boxes.cls, boxes.id):
                        cls = int(cls)
                        track_id = int(track_id)

                        # Filter only vehicles
                        if cls in VEHICLE_CLASSES:
                            counted_ids.add(track_id)
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(resized, f"ID {track_id}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw counter
                cv2.putText(resized, f"Vehicle Count: {len(counted_ids)}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                annotated_resized = cv2.resize(resized, (frame_width, frame_height))
                out.write(annotated_resized)

                # Update UI
                frame_count += 1
                progress = min(frame_count / total_frames, 1.0) if total_frames > 0 else 0
                progress_bar.progress(progress)
                status_text.text(
                    f"Processing frame {frame_count} / {total_frames} (Vehicles counted: {len(counted_ids)})")

                # Show every 5th frame in UI to prevent browser lag
                if frame_count % 5 == 0:
                    frame_window.image(cv2.cvtColor(annotated_resized, cv2.COLOR_BGR2RGB), channels="RGB")

            cap.release()
            out.release()

            progress_bar.empty()
            status_text.success(f"✅ Video processing complete! Total unique vehicles: {len(counted_ids)}")

            # Provide download button for the processed video
            with open(out_file.name, 'rb') as f:
                st.download_button(
                    label="Download Processed Video",
                    data=f,
                    file_name="howlingwolfs_output.mp4",
                    mime="video/mp4"
                )