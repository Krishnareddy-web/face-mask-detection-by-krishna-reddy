import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import threading
from detector import detect_and_predict_mask

class MaskDetectorApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Face Mask Detector")
        self.video_running = False
        self.cap = None

        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        btn_frame = tk.Frame(window)
        btn_frame.pack(fill=tk.X)
        tk.Button(btn_frame, text="Start Video", command=self.start_video).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Stop Video", command=self.stop_video).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Select Image", command=self.select_image).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Exit", command=window.quit).pack(side=tk.RIGHT)

    def start_video(self):
        if not self.video_running:
            self.cap = cv2.VideoCapture(0)
            self.video_running = True
            threading.Thread(target=self.video_loop).start()

    def video_loop(self):
        while self.video_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            boxes, preds = detect_and_predict_mask(frame)
            for (box, pred) in zip(boxes, preds):
                x, y, w, h = box
                mask_prob, no_mask_prob = pred
                if mask_prob > 0.99:
                    color = (0, 255, 0)
                    label = "Mask"
                elif mask_prob < 0.5:
                    color = (0, 0, 255)
                    label = "No Mask"
                else:
                    color = (0, 255, 255)
                    label = "Uncertain"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label}: {max(mask_prob, no_mask_prob)*100:.2f}%", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk

    def stop_video(self):
        if self.video_running:
            self.video_running = False
            if self.cap is not None:
                self.cap.release()
            self.canvas.delete("all")

    def select_image(self):
        self.stop_video()
        file_path = filedialog.askopenfilename()
        if file_path:
            image = cv2.imread(file_path)
            boxes, preds = detect_and_predict_mask(image)
            for (box, pred) in zip(boxes, preds):
                x, y, w, h = box
                mask_prob, no_mask_prob = pred
                if mask_prob > 0.99:
                    color = (0, 255, 0)
                    label = "Mask"
                elif mask_prob < 0.5:
                    color = (0, 0, 255)
                    label = "No Mask"
                else:
                    color = (0, 255, 255)
                    label = "Uncertain"
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, f"{label}: {max(mask_prob, no_mask_prob)*100:.2f}%", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            imgtk = ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk

if __name__ == "__main__":
    root = tk.Tk()
    app = MaskDetectorApp(root)
    root.mainloop()
