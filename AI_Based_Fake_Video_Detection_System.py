
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from transformers import pipeline


# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="DeepGuard AI",
    page_icon="🛡️",
    layout="wide"
)

MODEL_NAME = "dima806/deepfake_vs_real_image_detection"


# -------------------- LOAD AI MODEL --------------------

@st.cache_resource(show_spinner="Loading AI model for the first time...")
def load_detector():
    return pipeline(
        "image-classification",
        model=MODEL_NAME,
        device=-1
    )


# -------------------- EXTRACT VIDEO FRAMES --------------------

def extract_frames(video_bytes, suffix, max_frames=8):
    temp_path = None
    frames = []

    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False
        ) as temp_file:
            temp_file.write(video_bytes)
            temp_path = temp_file.name

        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            raise ValueError(
                "Video open nahi ho payi. Dusri MP4 video try karo."
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            raise ValueError("Video ke frames read nahi ho paaye.")

        frame_numbers = np.linspace(
            0,
            total_frames - 1,
            num=min(max_frames, total_frames),
            dtype=int
        )

        for frame_number in frame_numbers:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
            success, frame = cap.read()

            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                frames.append((int(frame_number), image))

        cap.release()

        if not frames:
            raise ValueError("Video se koi readable frame nahi mila.")

        return frames

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# -------------------- INTERPRET MODEL LABELS --------------------

def normalize_label(label):
    label = str(label).strip().lower()

    if "fake" in label or "deepfake" in label:
        return "Fake"

    if "real" in label or "genuine" in label or "authentic" in label:
        return "Real"

    # Never guess what LABEL_0 or LABEL_1 means.
    return None


def analyze_frame(detector, image):
    predictions = detector(image, top_k=None)

    real_score = None
    fake_score = None

    for item in predictions:
        label = normalize_label(item.get("label", ""))
        score = float(item.get("score", 0.0))

        if label == "Real":
            real_score = score
        elif label == "Fake":
            fake_score = score

    if real_score is None or fake_score is None:
        raise ValueError(
            "Model ke labels Real/Fake ke roop mein identify nahi hue. "
            "Model labels ko verify karna hoga."
        )

    return real_score, fake_score


# -------------------- FINAL VIDEO DECISION --------------------

def decide_result(real_avg, fake_avg, frame_results):
    fake_votes = sum(
        row["Prediction"] == "Fake" for row in frame_results
    )

    real_votes = sum(
        row["Prediction"] == "Real" for row in frame_results
    )

    total = len(frame_results)

    fake_ratio = fake_votes / total
    real_ratio = real_votes / total

    # Strong model scores AND agreement across sampled frames.
    if fake_avg >= 0.75 and fake_ratio >= 0.65:
        return "FAKE", fake_ratio, real_ratio

    if real_avg >= 0.75 and real_ratio >= 0.65:
        return "REAL", fake_ratio, real_ratio

    return "SUSPICIOUS", fake_ratio, real_ratio


# -------------------- USER INTERFACE --------------------

st.title("🛡️ DeepGuard AI")
st.subheader("AI-Assisted Video Screening System")

st.write(
    "Upload a video to analyze selected frames for visual signs "
    "of real or AI-generated content."
)

st.warning(
    "This is an experimental screening tool, not a definitive "
    "deepfake verification system. Results may be incorrect."
)

uploaded_video = st.file_uploader(
    "Upload a video",
    type=["mp4", "mov", "avi", "mkv"]
)

max_frames = st.slider(
    "Frames to analyze",
    min_value=4,
    max_value=16,
    value=8,
    help="More frames can take longer to analyze."
)

if uploaded_video is not None:
    st.video(uploaded_video.getvalue())

    if st.button("🔍 Analyze Video", type="primary"):
        try:
            with st.spinner("Extracting video frames..."):
                suffix = Path(uploaded_video.name).suffix or ".mp4"

                frames = extract_frames(
                    uploaded_video.getvalue(),
                    suffix,
                    max_frames
                )

            st.success(f"{len(frames)} frames extracted.")

            with st.spinner(
                "Loading model and analyzing frames. Please wait..."
            ):
                detector = load_detector()

                frame_results = []
                progress = st.progress(0)

                for index, (frame_number, image) in enumerate(frames):
                    real_score, fake_score = analyze_frame(
                        detector, image
                    )

                    if fake_score > real_score:
                        prediction = "Fake"
                    elif real_score > fake_score:
                        prediction = "Real"
                    else:
                        prediction = "Unclear"

                    frame_results.append({
                        "Frame": frame_number,
                        "Real score (%)": round(real_score * 100, 2),
                        "Fake score (%)": round(fake_score * 100, 2),
                        "Prediction": prediction
                    })

                    progress.progress(
                        (index + 1) / len(frames)
                    )

            real_avg = sum(
                row["Real score (%)"] for row in frame_results
            ) / len(frame_results) / 100

            fake_avg = sum(
                row["Fake score (%)"] for row in frame_results
            ) / len(frame_results) / 100

            result, fake_ratio, real_ratio = decide_result(
                real_avg,
                fake_avg,
                frame_results
            )

            st.divider()
            st.header("📊 Video Analysis Result")

            if result == "REAL":
                st.success("Result: REAL")
            elif result == "FAKE":
                st.error("Result: FAKE")
            else:
                st.warning("Result: SUSPICIOUS")

            st.caption(
                "This result reflects the model's visual assessment "
                "of sampled frames, not proof of authenticity."
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Average Real model score",
                f"{real_avg * 100:.2f}%"
            )

            col2.metric(
                "Average Fake model score",
                f"{fake_avg * 100:.2f}%"
            )

            st.subheader("📈 Real vs Fake Score Graph")

            chart_data = pd.DataFrame(
                {"Average model score (%)": [
                    real_avg * 100,
                    fake_avg * 100
                ]},
                index=["Real", "Fake"]
            )

            st.bar_chart(chart_data)

            st.subheader("🎞️ Frame-by-Frame Analysis")

            report_df = pd.DataFrame(frame_results)
            st.dataframe(report_df, use_container_width=True)

            st.subheader("🧠 Interpretation")

            st.write(
                f"- Frames leaning toward Fake: "
                f"{fake_ratio * 100:.1f}%"
            )
            st.write(
                f"- Frames leaning toward Real: "
                f"{real_ratio * 100:.1f}%"
            )

            if result == "SUSPICIOUS":
                st.info(
                    "The model did not meet the configured threshold "
                    "and frame-agreement requirements. This does not "
                    "automatically mean the video is fake."
                )

            report_text = (
                "DEEPGUARD AI - VIDEO SCREENING REPORT\n"
                "====================================\n"
                f"Video: {uploaded_video.name}\n"
                f"Frames analyzed: {len(frame_results)}\n"
                f"Result: {result}\n"
                f"Average Real model score: {real_avg * 100:.2f}%\n"
                f"Average Fake model score: {fake_avg * 100:.2f}%\n"
                f"Frames leaning Fake: {fake_ratio * 100:.1f}%\n"
                f"Frames leaning Real: {real_ratio * 100:.1f}%\n\n"
                "NOTE: Experimental model output; not proof of authenticity.\n"
            )

            st.download_button(
                "⬇️ Download Analysis Report",
                data=report_text,
                file_name="deepguard_video_report.txt",
                mime="text/plain"
            )

        except Exception as error:
            st.error(f"Analysis could not be completed: {error}")

            st.info(
                "Check your internet connection for the first model "
                "download. If the model-label error appears, send me "
                "the complete error message."
            )

st.divider()
st.caption("DeepGuard AI | AI-assisted video screening project")
#cd /d C:\Users\pc
# #conda activate deepguard
# #python -m streamlit run AI_Based_Fake_Video_Detection_System.py
