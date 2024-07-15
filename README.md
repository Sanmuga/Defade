# DEFADE - Deepfake Detection System

DEFADE is a real-time deepfake detection system designed to identify manipulated facial content in images and videos. Leveraging advanced deep learning techniques, DEFADE provides accurate detection and visual explainability to enhance trust and transparency.

## Features
- **Real-time Detection:** Analyze images and videos to detect deepfakes efficiently.
- **Advanced Models:** Utilizes PyTorch, MTCNN, and InceptionResnetV1 models for robust performance.
- **Visual Explainability:** Incorporates GradCAM for heatmap visualizations to explain detection results.
- **User-friendly Interface:** Streamlit-based interface for easy media upload and processing.

## Installation

### Prerequisites
- Python 3.7 or higher
- Pip (Python package installer)

### Clone the Repository
```bash
git clone https://github.com/yourusername/defade.git
cd defade
```

### Create a Virtual Environment
```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application
```bash
streamlit run Defade.py
```

### Interface
- **Image Deepfake Detection:** Upload an image and click 'Detect' to analyze and view results.
- **Video Deepfake Detection:** Upload a video and click 'Detect' to process and view frame-by-frame results.

## Technologies Used
- **Programming Languages:** Python
- **Libraries:** PyTorch, MTCNN, OpenCV, GradCAM, NumPy, PIL
- **Frameworks:** Streamlit
- **Models:** InceptionResnetV1 (pretrained on VGGFace2)

## Contributing
We welcome contributions to enhance DEFADE. Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any inquiries or issues, please contact sanmugavadivel23@gmail.com

---

Feel free to customize the details such as the GitHub repository URL, contact information, and any other specifics relevant to your project.
