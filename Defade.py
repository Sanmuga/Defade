# import streamlit as st
# import torch
# import torch.nn.functional as F
# from facenet_pytorch import MTCNN, InceptionResnetV1
# import cv2
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from PIL import Image
# import os

# # Check if CUDA is available
# DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# # Initialize MTCNN for face detection
# mtcnn = MTCNN(select_largest=False, post_process=False, device=DEVICE).eval()

# # Initialize the InceptionResnetV1 model for classification
# model = InceptionResnetV1(pretrained="vggface2", classify=True, num_classes=1, device=DEVICE)

# # Load the model checkpoint
# try:
#     checkpoint = torch.load("resnetinceptionv1_epoch_32.pth", map_location=DEVICE)
#     model.load_state_dict(checkpoint['model_state_dict'])
# except Exception as e:
#     st.error(f"Error loading model checkpoint: {e}")

# model.to(DEVICE).eval()

# # Preprocessing function
# def preprocess(face):
#     face = face.unsqueeze(0)  # Add batch dimension
#     face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)
#     face = face.to(DEVICE).to(torch.float32) / 255.0
#     return face

# # Prediction function for image
# def predict_image(input_image):
#     try:
#         if isinstance(input_image, str):
#             input_image = Image.open(input_image).convert('RGB')
        
#         face = mtcnn(input_image)
#         if face is None:
#             return "No face detected", None

#         face = preprocess(face)

#         prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy().astype('uint8')
#         face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()

#         target_layers = [model.block8.branch1[-1]]
#         cam = GradCAM(model=model, target_layers=target_layers)
#         targets = [ClassifierOutputTarget(0)]

#         grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
#         grayscale_cam = grayscale_cam[0, :]
#         visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
#         face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)

#         with torch.no_grad():
#             output = torch.sigmoid(model(face).squeeze(0))
#             real_prediction = (1 - output.item()) * 100
#             fake_prediction = output.item() * 100
            
#             confidences = {
#                 'real': real_prediction,
#                 'fake': fake_prediction
#             }
#         return confidences, face_with_mask
    
#     except Exception as e:
#         return f"Error in prediction: {e}", None

# # Face detection with OpenCV's Haar Cascade
# face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# def detect_bounding_box(vid):
#     gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
#     faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
#     for (x, y, w, h) in faces:
#         cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
#     return faces

# # Prediction function for video
# def predict_video(frame):
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     input_image = Image.fromarray(frame_rgb)

#     faces = detect_bounding_box(frame)  # Detect faces

#     fake_detected = False
#     for (x, y, w, h) in faces:
#         face_region = frame[y:y + h, x:x + w]  # Extract face region

#         # Perform face recognition on the extracted face region
#         input_face = Image.fromarray(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))
#         input_face = mtcnn(input_face)
#         if input_face is None:
#             continue

#         input_face = input_face.unsqueeze(0)  # Add the batch dimension
#         input_face = F.interpolate(input_face, size=(256, 256), mode="bilinear", align_corners=False)
#         input_face = input_face.to(DEVICE).to(torch.float32) / 255.0

#         target_layers = [model.block8.branch1[-1]]
#         cam = GradCAM(model=model, target_layers=target_layers)
#         targets = [ClassifierOutputTarget(0)]

#         grayscale_cam = cam(input_tensor=input_face, targets=targets, eigen_smooth=True)
#         grayscale_cam = grayscale_cam[0, :]
#         visualization = show_cam_on_image(input_face.squeeze(0).permute(1, 2, 0).cpu().detach().numpy(), grayscale_cam, use_rgb=True)

#         with torch.no_grad():
#             output = torch.sigmoid(model(input_face).squeeze(0))
#             prediction = "Fake" if output.item() < 0.5 else "Real"

#         if prediction == "Fake":
#             fake_detected = True
#             frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
#             frame = cv2.putText(frame, "Deep Fake Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#         else:
#             frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)
#             frame = cv2.putText(frame, "Real Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#     return frame, fake_detected

# # Process video function
# def process_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

#     fake_frames = []
#     frame_count = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         result_frame, fake_detected = predict_video(frame)
#         out.write(result_frame)
#         if fake_detected:
#             fake_frames.append(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
#         frame_count += 1

#     cap.release()
#     out.release()

#     if not fake_frames:
#         return 'output.mp4', "No fake frames detected."

#     fake_frames_dir = 'fake_frames'
#     if not os.path.exists(fake_frames_dir):
#         os.makedirs(fake_frames_dir)

#     for i, fake_frame in enumerate(fake_frames):
#         fake_frame_path = os.path.join(fake_frames_dir, f'fake_frame_{i}.png')
#         cv2.imwrite(fake_frame_path, fake_frame[:, :, ::-1])

#     return "FAKE",'output.mp4', fake_frames[3]

# # Streamlit UI
# def main():
#     st.title("DeFaDe - Deepfake Detection")
#     st.markdown("Choose whether to upload an image or a video for deepfake detection.")

#     choice = st.radio("Choose an option", ("Image", "Video"))

#     if choice == "Image":
#         st.markdown("### Deepfake Detection in Image")
#         uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

#         if uploaded_image is not None:
#             image = Image.open(uploaded_image)
#             st.image(image, caption="Uploaded Image", use_column_width=True)

#             if st.button("Detect"):
#                 result, explainability_image = predict_image(image)
#                 if explainability_image is not None:
#                     st.image(explainability_image, caption="Image with Explainability", use_column_width=True)

#                 st.markdown("### Prediction:")
#                 st.write(f"Real: {result['real']:.2f}%")
#                 st.write(f"Fake: {result['fake']:.2f}%")
#                 st.bar_chart(result)

#     elif choice == "Video":
#         st.markdown("### Deepfake Detection in Video")
#         uploaded_video = st.file_uploader("Upload a video", type=["mp4"])

#         if uploaded_video is not None:
#             st.video(uploaded_video)

#             if st.button("Detect"):
#                 video_path = "temp_video.mp4"
#                 with open(video_path, 'wb') as f:
#                     f.write(uploaded_video.read())
                
#                 output_path, fake_frames = process_video(video_path)
#                 st.success(f"Processed video saved as {output_path}")

#                 if fake_frames:
#                     st.markdown("### Detected Fake Frames")
#                     for idx, frame in enumerate(fake_frames):
#                         st.image(frame, caption=f"Fake Frame {idx+1}", use_column_width=True)

# # Run the app
# if __name__ == "__main__":
#     main()


import streamlit as st
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import os

# Check if CUDA is available
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Initialize MTCNN for face detection
mtcnn = MTCNN(select_largest=False, post_process=False, device=DEVICE).eval()

# Initialize the InceptionResnetV1 model for classification
model = InceptionResnetV1(pretrained="vggface2", classify=True, num_classes=1, device=DEVICE)

# Load the model checkpoint
try:
    checkpoint = torch.load("resnetinceptionv1_epoch_32.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
except Exception as e:
    st.error(f"Error loading model checkpoint: {e}")

model.to(DEVICE).eval()

# Preprocessing function
def preprocess(face):
    face = face.unsqueeze(0)  # Add batch dimension
    face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)
    face = face.to(DEVICE).to(torch.float32) / 255.0
    return face

# Prediction function for image
def predict_image(input_image):
    try:
        if isinstance(input_image, str):
            input_image = Image.open(input_image).convert('RGB')
        
        face = mtcnn(input_image)
        if face is None:
            return "No face detected", None

        face = preprocess(face)

        prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy().astype('uint8')
        face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()

        target_layers = [model.block8.branch1[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(0)]

        grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
        face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)

        with torch.no_grad():
            output = torch.sigmoid(model(face).squeeze(0))
            real_prediction = (1 - output.item()) * 100
            fake_prediction = output.item() * 100
            
            confidences = {
                'real': real_prediction,
                'fake': fake_prediction
            }
        return confidences, face_with_mask
    
    except Exception as e:
        return f"Error in prediction: {e}", None

# Face detection with OpenCV's Haar Cascade
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

# Prediction function for video
def predict_video(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = Image.fromarray(frame_rgb)

    faces = detect_bounding_box(frame)  # Detect faces

    fake_detected = False
    for (x, y, w, h) in faces:
        face_region = frame[y:y + h, x:x + w]  # Extract face region

        # Perform face recognition on the extracted face region
        input_face = Image.fromarray(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))
        input_face = mtcnn(input_face)
        if input_face is None:
            continue

        input_face = input_face.unsqueeze(0)  # Add the batch dimension
        input_face = F.interpolate(input_face, size=(256, 256), mode="bilinear", align_corners=False)
        input_face = input_face.to(DEVICE).to(torch.float32) / 255.0

        target_layers = [model.block8.branch1[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(0)]

        grayscale_cam = cam(input_tensor=input_face, targets=targets, eigen_smooth=True)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(input_face.squeeze(0).permute(1, 2, 0).cpu().detach().numpy(), grayscale_cam, use_rgb=True)

        with torch.no_grad():
            output = torch.sigmoid(model(input_face).squeeze(0))
            prediction = "Fake" if output.item() < 0.5 else "Real"

        if prediction == "Fake":
            fake_detected = True
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
            frame = cv2.putText(frame, "Deep Fake Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)
            frame = cv2.putText(frame, "Real Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame, fake_detected

# Process video function
def process_video(video_path, frame_skip=5, target_resolution=(640, 480)):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, target_resolution)

    fake_frames = []
    frame_count = 0
    progress_text = "Processing video. Please wait."
    my_bar = st.progress(0, text=progress_text)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            frame = cv2.resize(frame, target_resolution)
            result_frame, fake_detected = predict_video(frame)
            out.write(result_frame)
            if fake_detected:
                fake_frames.append(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
            processed_frames += 1

        frame_count += 1
        progress_percent = processed_frames / total_frames * 100
        my_bar.progress(int(progress_percent), text=progress_text)

    cap.release()
    out.release()
    my_bar.empty()

    if not fake_frames:
        return 'output.mp4', None

    fake_frames_dir = 'fake_frames'
    if not os.path.exists(fake_frames_dir):
        os.makedirs(fake_frames_dir)

    for i, fake_frame in enumerate(fake_frames):
        fake_frame_path = os.path.join(fake_frames_dir, f'fake_frame_{i}.png')
        cv2.imwrite(fake_frame_path, fake_frame[:, :, ::-1])

    return 'output.mp4', fake_frames[:3]

# Streamlit UI
def main():
    st.title("DeFaDe - Deepfake Detection")
    st.markdown("Choose whether to upload an image or a video for deepfake detection.")

    choice = st.radio("Choose an option", ("Image", "Video"))

    if choice == "Image":
        st.markdown("### Deepfake Detection in Image")
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Detect"):
                result, explainability_image = predict_image(image)
                if explainability_image is not None:
                    st.image(explainability_image, caption="Image with Explainability", use_column_width=True)

                st.markdown("### Prediction:")
                st.write(f"Real: {result['real']:.2f}%")
                st.write(f"Fake: {result['fake']:.2f}%")
                st.bar_chart(result)

    elif choice == "Video":
        st.markdown("### Deepfake Detection in Video")
        uploaded_video = st.file_uploader("Upload a video", type=["mp4"])

        if uploaded_video is not None:
            st.video(uploaded_video)

            if st.button("Detect"):
                video_path = "temp_video.mp4"
                with open(video_path, 'wb') as f:
                    f.write(uploaded_video.read())

                st.text("Choose frame processing options:")
                frame_skip = st.slider("Process every nth frame (e.g., 5):", 1, 10, 5)
                target_width = st.slider("Target width (e.g., 640):", 320, 1280, 640)
                target_height = st.slider("Target height (e.g., 480):", 240, 720, 480)
                target_resolution = (target_width, target_height)

                output_path, fake_frames = process_video(video_path, frame_skip, target_resolution)
                st.success(f"Processed video saved as {output_path}")

                if fake_frames:
                    st.markdown("### Detected Fake Frames")
                    for idx, frame in enumerate(fake_frames):
                        st.image(frame, caption=f"Fake Frame {idx+1}", use_column_width=True)

# Run the app
if __name__ == "__main__":
    main()


