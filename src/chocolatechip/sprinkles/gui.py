import os
import cv2
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import time
import yaml
import sys
import pyperclip

class VideoPlayer:
    def __init__(self, root, video_files):
        self.root = root
        self.video_files = video_files
        self.current_video_index = 0
        self.video = None
        self.frame = None
        self.paused = False
        self.frame_delay = 75  # Delay for 10 FPS
        self.frame_id = None  # ID for frame update loop
        self.start_time = 0

        # Load the YAML file
        with open('sprinkles.yaml', 'r') as f:
            self.yaml_data = yaml.load(f, Loader=yaml.FullLoader)

        # Setting up the GUI
        self.filename_label = tk.Label(root, text="", font=("Helvetica", 12))
        self.filename_label.pack()

        self.copy_button = tk.Button(root, text="Copy Filename to Clipboard", command=self.copy_filename_to_clipboard)
        self.copy_button.pack()

        self.label = tk.Label(root)
        self.label.pack()

        self.progress = ttk.Progressbar(root, orient="horizontal", length=640, mode="determinate")
        self.progress.pack()

        self.instructions_label = tk.Label(root, text="N means dangerous, M means harmless, Z for previous, X for next", font=("Helvetica", 10))
        self.instructions_label.pack()

        self.status_label = tk.Label(root, text="", font=("Helvetica", 10))
        self.status_label.pack()
        
        self.timestamp_label = tk.Label(root, text="00:00 / 00:00", font=("Helvetica", 10))
        self.timestamp_label.pack()

        self.root.bind("<Left>", self.skip_backward)
        self.root.bind("<Right>", self.skip_forward)
        self.root.bind("<n>", self.mark_dangerous)
        self.root.bind("<m>", self.mark_harmless)
        self.root.bind("<z>", self.previous_video)
        self.root.bind("<x>", self.next_video_without_change)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.play_video()

    def copy_filename_to_clipboard(self):
        video_path = self.video_files[self.current_video_index]
        pyperclip.copy(os.path.basename(video_path))
        # messagebox.showinfo("Copied", "Filename copied to clipboard")

    def play_video(self):
        video_path = self.video_files[self.current_video_index]
        self.video = cv2.VideoCapture(video_path)
        video_name = os.path.basename(video_path)
        self.filename_label.config(text=video_name)  # Update filename label
        self.update_status_label(video_name)
        self.fps = 10  # Hardcoded FPS
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count = 0  # Initialize frame count
        self.start_time = time.time()
        self.update_progress()
        self.update_timestamp()
        self.show_frame()

    def show_frame(self):
        ret, frame = self.video.read()
        if ret:
            self.frame = frame
            # Resize frame to fit in a smaller window
            frame = cv2.resize(frame, (800, 600))
            cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv_image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

            self.frame_count += 1
            self.frame_id = self.root.after(self.frame_delay, self.show_frame)
        else:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_count = 0
            self.show_frame()

    def stop_frame_update(self):
        if self.frame_id is not None:
            self.root.after_cancel(self.frame_id)
            self.frame_id = None

    def update_progress(self):
        current_frame = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
        progress_value = (current_frame / self.total_frames) * 100
        self.progress["value"] = progress_value
        self.root.after(100, self.update_progress)

    def update_timestamp(self):
        current_time = int(self.frame_count / self.fps)
        total_time = int(self.total_frames / self.fps)
        current_minutes, current_seconds = divmod(current_time, 60)
        total_minutes, total_seconds = divmod(total_time, 60)
        self.timestamp_label.config(text=f"{current_minutes:02}:{current_seconds:02} / {total_minutes:02}:{total_seconds:02}")
        self.root.after(1000, self.update_timestamp)

    def update_status_label(self, video_name):
        status = self.yaml_data.get(video_name, "unknown")
        if status == "dangerous":
            self.status_label.config(text=f"The current video is marked as: {status}", fg="red")
        else:
            self.status_label.config(text=f"The current video is marked as: {status}", fg="black")
    
    def skip_backward(self, event):
        current_time = self.video.get(cv2.CAP_PROP_POS_MSEC)
        self.video.set(cv2.CAP_PROP_POS_MSEC, max(0, current_time - 3000))
        self.update_progress()
        self.update_timestamp()
    
    def skip_forward(self, event):
        current_time = self.video.get(cv2.CAP_PROP_POS_MSEC)
        self.video.set(cv2.CAP_PROP_POS_MSEC, current_time + 3000)
        self.update_progress()
        self.update_timestamp()
    
    def mark_dangerous(self, event):
        video_path = self.video_files[self.current_video_index]
        video_name = os.path.basename(video_path)
        self.yaml_data[video_name] = "dangerous"
        self.save_yaml()
        self.update_status_label(video_name)
        self.next_video()
    
    def mark_harmless(self, event):
        video_path = self.video_files[self.current_video_index]
        video_name = os.path.basename(video_path)
        self.yaml_data[video_name] = "harmless"
        self.save_yaml()
        self.update_status_label(video_name)
        self.next_video()

    def save_yaml(self):
        with open('sprinkles.yaml', 'w') as f:
            yaml.dump(self.yaml_data, f)
    
    def next_video(self):
        self.stop_frame_update()
        self.video.release()
        self.current_video_index += 1
        if self.current_video_index >= len(self.video_files):
            self.current_video_index = 0
        self.play_video()
    
    def next_video_without_change(self, event):
        self.stop_frame_update()
        self.video.release()
        self.current_video_index += 1
        if self.current_video_index >= len(self.video_files):
            self.current_video_index = 0
        self.play_video()
    
    def previous_video(self, event):
        self.stop_frame_update()
        self.video.release()
        self.current_video_index -= 1
        if self.current_video_index < 0:
            self.current_video_index = len(self.video_files) - 1
        self.play_video()

    def on_closing(self):
        self.stop_frame_update()
        self.root.destroy()
        sys.exit()

def get_video_files():
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    return [f for f in os.listdir('.') if f.endswith(video_extensions)]

def create_yaml_file(video_files):
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    start_range = simpledialog.askstring("Input", "Enter the start range (e.g., 000):", parent=root)
    end_range = simpledialog.askstring("Input", "Enter the end range (e.g., 1201):", parent=root)

    if start_range is None or end_range is None:
        messagebox.showerror("Error", "You must enter both start and end ranges.")
        return None

    start_range = int(start_range)
    end_range = int(end_range)
    valid_videos = {}

    for video in video_files:
        video_number = int(video.split('_')[1].split('.')[0])
        if start_range <= video_number <= end_range:
            valid_videos[video] = "dangerous"

    with open('sprinkles.yaml', 'w') as f:
        yaml.dump(valid_videos, f)

    return [video for video in valid_videos]

def load_yaml_file():
    with open('sprinkles.yaml', 'r') as f:
        valid_videos = yaml.load(f, Loader=yaml.FullLoader)
    return [video for video in valid_videos]

def main():
    root = tk.Tk()
    root.title("Video Player")
    video_files = get_video_files()

    if not video_files:
        messagebox.showerror("Error", "No video files found in the current directory.")
        return

    if not os.path.exists('sprinkles.yaml'):
        video_files = create_yaml_file(video_files)
        if video_files is None:
            return
    else:
        video_files = load_yaml_file()

    VideoPlayer(root, video_files)
    root.mainloop()

if __name__ == "__main__":
    main()
