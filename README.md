**😷 Face Mask Detection System (Image + Webcam)**

A deep learning–based Face Mask Detection project built using PyTorch, capable of detecting whether a person is wearing a mask or not, using both:
📷 Static Images
🎥 Real-Time Webcam (Google Colab)

**📌 Features**

Binary classification: with_mask vs without_mask
Transfer learning using MobileNetV2
Image-based prediction
Real-time webcam photo capture and prediction (Colab)
Clean dataset handling using ImageFolder

**🧠 Model Architecture**

Base model: MobileNetV2 (pretrained on ImageNet)
Final layer: Modified for 2 classes
Loss: CrossEntropyLoss
Optimizer: Adam
Input size: 128 × 128 RGB images

**📁 Dataset Structure**

The project expects the dataset in this format:

dataset/
│
├── with_mask/
│   ├── img1.jpg
│   ├── img2.jpg
│
├── without_mask/
│   ├── img3.jpg
│   ├── img4.jpg


**⚠️ Important:**
Folders like annotations/ or images/ must be removed before training, otherwise ImageFolder will throw errors.

⚙️ Installation & Setup (Google Colab)
pip install torch torchvision opencv-python pillow matplotlib kaggle


Upload your kaggle.json file and run:

mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

🚀 Training the Model

Load dataset using ImageFolder
Split into train/test (80/20)
Train MobileNetV2 for 5 epochs
Save model as:

mask_detector.pth

🖼️ Simple Image Mask Detection
📌 How It Works

Load trained model
Pass image path
Get prediction: with_mask or without_mask
✅ Example Code
label = predict("dataset/with_mask/sample.jpg")
print("Prediction:", label)

**📊 Output**

The image is displayed with its predicted label.

**🎥 Webcam Mask Detection (Google Colab)**

Since Colab cannot stream live video, the webcam mode works by:
Opening webcam
Capturing a photo
Saving it as photo.jpg
Running mask detection on the captured image

**📷 Step A:** Capture Image from Webcam
photo_path = take_photo()
print("Photo saved at:", photo_path)

A Capture Photo button appears — click it to take a picture.

**🤖 Step B:** Predict Mask from Captured Photo
label, _ = predict(photo_path)

from PIL import Image
import matplotlib.pyplot as plt

plt.imshow(Image.open(photo_path))
plt.title(f"Prediction: {label.upper()}")
plt.axis("off")
plt.show()

**🛑 How to Stop Webcam?**

The webcam automatically stops after photo capture
No manual stop required
Just run the prediction cell after capture

**✅ Output Labels**

Label	Meaning
with_mask	Person is wearing a mask
without_mask	Person is not wearing a mask
🧪 Common Errors & Fixes
❌ FileNotFoundError: Found no valid file for class annotations
✔ Fix:
shutil.rmtree("dataset/annotations", ignore_errors=True)
shutil.rmtree("dataset/images", ignore_errors=True)
❌ NameError: predict not defined
✔ Fix:
Make sure the predict function cell is run before webcam prediction.

**📌 Technologies Used**

Python
PyTorch
Torchvision
OpenCV
PIL
Matplotlib
Google Colab Webcam API

**📄 License**

This project is for educational and research purposes.

⭐ If you like this project

Give it a ⭐ on GitHub
Fork it
Improve it with face detection + bounding boxes
