# 🍎 Fruit Detection (YOLOv8)

Real-time fruit detection powered by **YOLOv8**, using a **custom fruit dataset** split into **train / val / test**.  
This project is built for **quick deployment**, **easy testing**, and **clean reproducibility**.

---

## 🚀 Features

- 🧠 **YOLOv8n model** trained on a custom fruit dataset (Roboflow-based)
- 🧩 **Custom Dataset Split**: manually divided into train / validation / test sets
- 🎯 **Pretrained Weights**: run directly using `best.pt` without retraining
- 💻 **Simple Interface**: only one Python file to run detection
- 📄 **Essay Report**: full report provided separately (Google Drive)

---

## 📄 Essay Report

The full essay report has been uploaded to Google Drive.  
You can download it here:

---

## 🍉 Dataset Overview

The dataset contains **6 fruit classes** used for object detection:

| Class | Fruit | Description |
|------|------|-------------|
| 🍍 | Pineapple | Tropical fruit with spiky skin and sweet yellow flesh |
| 🍒 | Cherry | Small red fruit often appearing in pairs |
| 🥭 | Mango | Yellow-orange fruit with smooth skin and sweet aroma |
| 🍑 | Plum | Round fruit with smooth skin, purple or red when ripe |
| 🍅 | Tomato | Red juicy fruit often mistaken for a vegetable |
| 🍉 | Watermelon | Large green fruit with red interior and black seeds |

---

## 🗂 Project Structure
<img width="1167" height="451" alt="image" src="https://github.com/user-attachments/assets/aa629c3e-fe28-4566-be2e-2f5a69e94974" />


---

## ⚙️ Installation

### 1️⃣ Clone the repository

git clone https://github.com/Pfisthis2004/Fruit-Detection.git
cd Fruit-Detection

### 2️⃣ (Optional) Create virtual environment & install dependencies

Create a virtual environment (recommended):

python -m venv venv
### 3️⃣ Install dependencies 
pip install -r requirements.txt
## ▶️ Run Detection

### 🧩 Option 1 — Detect via Script
Run the detection script directly:

python program.py

Make sure your working directory includes:

weights/best.pt
dataset_fruits/data.yaml
The program loads the YOLOv8 model and runs inference directly.
### 🧠 Option 2 — Run in Spyder (Recommended for GUI)

  1.Open Anaconda Navigator

  2.Launch Spyder

  3.Open program.py

  4.Press Run (F5) to start detection
### 🧠 Model Details

Model: best.pt (trained YOLOv8n)
Framework: Ultralytics YOLOv8 (Python)
Dataset: Custom split version of Roboflow fruit dataset
Train / Val / Test ratio: Defined manually in README.dataset.md
## 📸 Preview
<img width="1487" height="905" alt="image" src="https://github.com/user-attachments/assets/17a72ad3-1276-4b8f-bc9f-875c476a6bf3" />

##🧾 License
This project is licensed under the MIT License.

