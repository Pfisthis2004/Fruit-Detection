import tkinter as tk
from tkinter import filedialog, Toplevel, ttk
from PIL import Image, ImageTk, ImageFont, ImageDraw
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import os

# ================== LOAD MODEL ==================
model = YOLO("runs/detect/train3/weights/best.pt")

#GLOBAL 
cap = None
running = False
last_results = []   # kết quả detect gần nhất [(label, conf, (x1,y1,x2,y2)), ...]
all_results = []    # lưu tất cả kết quả qua nhiều lần detect
frame_count = 0     # đếm số frame để vẽ biểu đồ

# DETECT & DISPLAY 
def detect_and_display(frame):
    global last_results, all_results, frame_count
    last_results = []  
    frame_count += 1

    results = model.predict(frame, imgsz=640, conf=0.5)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = result.names[cls]

            # Lưu chi tiết
            last_results.append((label, conf, (x1, y1, x2, y2)))
            all_results.append((frame_count, label, conf, (x1, y1, x2, y2)))

            # Vẽ khung
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Vẽ nhãn
            font = ImageFont.truetype("arial.ttf", 20)
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            draw.text((x1, y1 - 25), f"{label} {conf:.2f}", font=font, fill=(255, 0, 0))
            frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    return frame

#CAMERA
def start_camera():
    global cap, running
    cap = cv2.VideoCapture(0)
    running = True
    update_frame()

def stop_camera():
    global cap, running
    running = False
    if cap is not None:
        cap.release()
        cap = None
    label_display.config(image="")
    label_display.image = None

def update_frame():
    global cap, running
    if running and cap is not None:
        ret, frame = cap.read()
        if ret:
            frame = detect_and_display(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img_pil)
            label_display.config(image=imgtk)
            label_display.image = imgtk
        label_display.after(20, update_frame)

#OPEN IMAGE 
def open_image():
    global running, cap
    if running:
        stop_camera()

    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if not file_path:
        return

    img = cv2.imread(file_path)
    img = detect_and_display(img)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    imgtk = ImageTk.PhotoImage(image=img_pil)

    label_display.config(image=imgtk)
    label_display.image = imgtk

#ANALYZE 
analyze_window = None  # lưu cửa sổ phân tích
analyze_frame = None   # frame chứa nội dung phân tích

def analyze_results():
    global last_results, all_results, analyze_window, analyze_frame
    csv_path = r"C:\Users\LEGION\OneDrive\Documents\Python Scripts\TH1\runs\detect\train3\results.csv"

   
    if analyze_window is None or not tk.Toplevel.winfo_exists(analyze_window):
        analyze_window = Toplevel(root)
        analyze_window.title("Phân tích kết quả")
        analyze_window.geometry("1200x800")

        analyze_frame = tk.Frame(analyze_window)
        analyze_frame.pack(fill="both", expand=True)

   
    for widget in analyze_frame.winfo_children():
        widget.destroy()
    plt.close('all')

  
    if not last_results and not os.path.exists(csv_path):
        tk.Label(analyze_frame, text="Không có dữ liệu để phân tích!",
                 font=("Arial", 14), fg="red").pack(pady=20)
        return

    notebook = ttk.Notebook(analyze_frame)
    notebook.pack(fill="both", expand=True)

    #TAB DETECTION 
    tab_detect = ttk.Frame(notebook)
    notebook.add(tab_detect, text="Detection")

    if last_results:
        cols = ("Tên đối tượng", "Confidence", "Tọa độ (x1,y1,x2,y2)")
        tree = ttk.Treeview(tab_detect, columns=cols, show="headings")
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=200 if col != "Confidence" else 100, anchor="center")
        for (label, conf, (x1, y1, x2, y2)) in last_results:
            tree.insert("", "end", values=(label, f"{conf:.2f}", f"({x1},{y1},{x2},{y2})"))
        tree.pack(fill="x", padx=10, pady=10)

    fig1, axs = plt.subplots(1, 3, figsize=(12, 4))

    if all_results:
        
        frame_object_counts = {}
        for (f, _, _, _) in all_results:
            frame_object_counts[f] = frame_object_counts.get(f, 0) + 1
        axs[0].plot(list(frame_object_counts.keys()), list(frame_object_counts.values()), marker="o")
        axs[0].set_title("Tổng số object qua thời gian")
        axs[0].set_xlabel("Frame")
        axs[0].set_ylabel("Số object")

        
        conf_dict = {}
        for (f, _, conf, _) in all_results:
            conf_dict.setdefault(f, []).append(conf)
        avg_conf = [np.mean(v) for v in conf_dict.values()]
        axs[1].plot(list(conf_dict.keys()), avg_conf, color="orange", marker="o")
        axs[1].set_title("Confidence trung bình")
        axs[1].set_xlabel("Frame")
        axs[1].set_ylabel("Confidence")

       
        heatmap = np.zeros((480, 640))
        for (_, _, _, (x1, y1, x2, y2)) in all_results:
            x1 = max(0, min(639, x1))
            x2 = max(0, min(639, x2))
            y1 = max(0, min(479, y1))
            y2 = max(0, min(479, y2))
            heatmap[y1:y2, x1:x2] += 1
        axs[2].imshow(heatmap, cmap="hot", interpolation="nearest")
        axs[2].set_title("Heatmap vị trí object")

    plt.tight_layout()
    canvas1 = FigureCanvasTkAgg(fig1, master=tab_detect)
    canvas1.draw()
    canvas1.get_tk_widget().pack(fill="both", expand=True)

    #TAB TRAINING 
    if os.path.exists(csv_path):
        tab_train = ttk.Frame(notebook)
        notebook.add(tab_train, text="Training")

        df = pd.read_csv(csv_path)

        fig2, axs2 = plt.subplots(1, 3, figsize=(12, 4))

        
        axs2[0].plot(df["epoch"], df["train/box_loss"], label="train_box")
        axs2[0].plot(df["epoch"], df["val/box_loss"], label="val_box")
        axs2[0].set_title("Box Loss")
        axs2[0].legend()

      
        if "metrics/mAP50(B)" in df.columns:
            axs2[1].plot(df["epoch"], df["metrics/mAP50(B)"], color="green")
            axs2[1].set_title("mAP@0.5 theo epoch")

      
        if "metrics/precision(B)" in df.columns and "metrics/recall(B)" in df.columns:
            axs2[2].plot(df["epoch"], df["metrics/precision(B)"], label="Precision")
            axs2[2].plot(df["epoch"], df["metrics/recall(B)"], label="Recall")
            axs2[2].legend()
            axs2[2].set_title("Precision vs Recall")

        plt.tight_layout()
        canvas2 = FigureCanvasTkAgg(fig2, master=tab_train)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill="both", expand=True)


#478MAIN UI 
root = tk.Tk()
root.title("Nhận diện bằng Camera / Ảnh")
root.geometry("1200x700")
root.configure(bg="#ECEFF1")

control_frame = tk.Frame(root, width=220, bg="#CFD8DC")
control_frame.pack(side="left", fill="y", padx=10, pady=10)

title = tk.Label(control_frame, text="🎥 Nhận diện YOLO", 
                 bg="#CFD8DC", fg="#212121", 
                 font=("Arial", 16, "bold"))
title.pack(pady=20)

btn_start = tk.Button(control_frame, text="Bật Camera", command=start_camera,
                      bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), relief="flat")
btn_start.pack(pady=10, fill="x")

btn_stop = tk.Button(control_frame, text="Tắt Camera", command=stop_camera,
                     bg="#F44336", fg="white", font=("Arial", 12, "bold"), relief="flat")
btn_stop.pack(pady=10, fill="x")

btn_image = tk.Button(control_frame, text="Chọn Ảnh", command=open_image,
                      bg="#2196F3", fg="white", font=("Arial", 12, "bold"), relief="flat")
btn_image.pack(pady=10, fill="x")

btn_analyze = tk.Button(control_frame, text="Phân tích kết quả", command=analyze_results,
                        bg="#FF9800", fg="white", font=("Arial", 12, "bold"), relief="flat")
btn_analyze.pack(pady=10, fill="x")

display_frame = tk.Frame(root, bg="black")
display_frame.pack(side="right", expand=True, fill="both")

label_display = tk.Label(display_frame, bg="black")
label_display.pack(expand=True)

root.mainloop()
