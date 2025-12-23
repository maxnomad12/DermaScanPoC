# app3.py
import streamlit as st
import cv2
import numpy as np
import sys
import os
import traceback

st.set_page_config(layout="wide", page_title="DermaScan PoC")

COLOR_HIGH = (255, 0, 0)
COLOR_MEDIUM = (255, 255, 0)
COLOR_LOW = (0, 255, 0)


def get_resource_path(relative_path):
    base_path = getattr(sys, "_MEIPASS", None)
    if base_path:
        base = base_path
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, relative_path)


def analyze_pigmentation(uploaded_file):
    try:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        if file_bytes.size == 0:
            return None, None, "Error: Uploaded file seems empty."

        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None, None, "Error: Could not decode image. Try another file."

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # try packaged cascade (in _MEIPASS), otherwise fallback to cv2.data
        xml_file = "haarcascade_frontalface_default.xml"
        cascade_path_candidate = get_resource_path(os.path.join("cv2", "data", "haarcascades", xml_file))
        if os.path.exists(cascade_path_candidate):
            cascade_path = cascade_path_candidate
        else:
            cascade_path = os.path.join(cv2.data.haarcascades, xml_file)

        print(f"[DermaScan] Using cascade: {cascade_path}", file=sys.stderr)

        face_cascade = cv2.CascadeClassifier(cascade_path)
        # CascadeClassifier.empty() exists; True means load failed
        if face_cascade.empty():
            return img_rgb, None, f"Error: Could not load Haar Cascade from path: {cascade_path}"

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        if len(faces) == 0:
            return img_rgb, None, "Face not detected. Please try another photo."

        pigmentation_data = {"high": [], "medium": [], "low": []}

        for (x, y, w, h) in faces:
            y0, y1 = max(0, y), min(gray.shape[0], y + h)
            x0, x1 = max(0, x), min(gray.shape[1], x + w)
            roi_gray = gray[y0:y1, x0:x1]

            if roi_gray.size == 0:
                continue

            try:
                pigmentation_mask = cv2.adaptiveThreshold(
                    roi_gray,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    11,
                    2,
                )
            except Exception as e:
                print("[DermaScan] adaptiveThreshold failed:", e, file=sys.stderr)
                continue

            kernel = np.ones((2, 2), np.uint8)
            pigmentation_mask = cv2.morphologyEx(pigmentation_mask, cv2.MORPH_OPEN, kernel)

            contours_result = cv2.findContours(pigmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # compatibility across OpenCV versions
            contours = contours_result[0] if len(contours_result) == 2 else contours_result[1]

            for cnt in contours:
                area = cv2.contourArea(cnt)
                # filter out too small and too large artifacts
                if 5 < area < 500:
                    # shift contour coords from ROI -> full image
                    cnt_shifted = cnt.copy()
                    # ensure integer dtype to avoid warnings
                    cnt_shifted = cnt_shifted.astype(np.int32)
                    cnt_shifted[:, :, 0] = cnt_shifted[:, :, 0] + x0
                    cnt_shifted[:, :, 1] = cnt_shifted[:, :, 1] + y0

                    # mask in ROI coordinates to compute mean brightness
                    mask = np.zeros(roi_gray.shape, np.uint8)
                    try:
                        cv2.drawContours(mask, [cnt], -1, 255, -1)
                        mean_val = cv2.mean(roi_gray, mask=mask)[0]
                    except Exception as e:
                        print("[DermaScan] contour->mean failed:", e, file=sys.stderr)
                        continue

                    if mean_val < 80:
                        pigmentation_data["high"].append(cnt_shifted)
                    elif mean_val < 120:
                        pigmentation_data["medium"].append(cnt_shifted)
                    else:
                        pigmentation_data["low"].append(cnt_shifted)

        return img_rgb, pigmentation_data, "Success"

    except Exception as ex:
        traceback.print_exc(file=sys.stderr)
        return None, None, f"Unexpected error during analysis: {ex}"


def draw_masks(original_img, data, show_high, show_medium, show_low):
    if original_img is None:
        return None

    overlay = original_img.copy()
    try:
        if show_high and data.get("high"):
            cv2.drawContours(overlay, data["high"], -1, COLOR_HIGH, -1)
        if show_medium and data.get("medium"):
            cv2.drawContours(overlay, data["medium"], -1, COLOR_MEDIUM, -1)
        if show_low and data.get("low"):
            cv2.drawContours(overlay, data["low"], -1, COLOR_LOW, -1)
    except Exception as e:
        print("[DermaScan] draw_masks error:", e, file=sys.stderr)

    try:
        result = cv2.addWeighted(original_img, 0.7, overlay, 0.3, 0)
    except Exception as e:
        print("[DermaScan] addWeighted error:", e, file=sys.stderr)
        result = original_img
    return result


# UI
st.title("🧬 Facial Pigmentation Analysis (PoC)")
st.write("Upload a portrait photo to detect pigmentation areas. Use the sidebar to filter results.")

st.sidebar.header("Display Filters")
show_high = st.sidebar.checkbox("🔴 High Severity", value=True)
show_medium = st.sidebar.checkbox("🟡 Medium Severity", value=True)
show_low = st.sidebar.checkbox("🟢 Low Severity", value=True)

st.sidebar.markdown("---")
st.sidebar.header("Legend")
st.sidebar.markdown("🔴 **High:** Deep pigmentation")
st.sidebar.markdown("🟡 **Medium:** Moderate pigmentation")
st.sidebar.markdown("🟢 **Low:** Superficial pigmentation")

uploaded_file = st.file_uploader("Choose an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner("Analyzing image structure..."):
        original, data, status = analyze_pigmentation(uploaded_file)

    if data is not None and original is not None:
        processed_image = draw_masks(original, data, show_high, show_medium, show_low)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            # use_column_width is the correct parameter
            st.image(original, use_column_width=True)
        with col2:
            st.subheader("Analysis Result")
            st.image(processed_image, use_column_width=True)
        st.success(
            f"Analysis complete. Found: {len(data.get('high', []))} High, "
            f"{len(data.get('medium', []))} Medium, {len(data.get('low', []))} Low regions."
        )
    else:
        st.error(status or "Analysis failed. See terminal/log for details.")
