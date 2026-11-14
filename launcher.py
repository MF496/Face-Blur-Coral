"""Face Blur - small launcher window for face_blur.py."""
import glob
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "face_blur.py")


def list_cameras():
    """Return [(number, label)] for /dev/videoN devices."""
    cams = []
    for dev in sorted(glob.glob("/dev/video*"),
                      key=lambda p: int(re.sub(r"\D", "", p) or 0)):
        n = int(re.sub(r"\D", "", dev) or 0)
        name = ""
        try:
            with open(f"/sys/class/video4linux/video{n}/name") as f:
                name = f.read().strip()
        except OSError:
            pass
        cams.append((n, f"/dev/video{n}  {name}".rstrip()))
    return cams


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Blur")
        self.resizable(False, False)
        self.proc = None
        self.lines = queue.Queue()

        pad = {"padx": 8, "pady": 4}
        body = ttk.Frame(self, padding=10)
        body.grid(sticky="nsew")

        # --- source ---
        src = ttk.LabelFrame(body, text="Source", padding=8)
        src.grid(row=0, column=0, sticky="ew", **pad)
        self.mode = tk.StringVar(value="camera")
        ttk.Radiobutton(src, text="Live camera", variable=self.mode,
                        value="camera", command=self.update_state
                        ).grid(row=0, column=0, sticky="w")
        self.cam_box = ttk.Combobox(src, state="readonly", width=38)
        self.cam_box.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(src, text="Refresh", command=self.refresh_cameras
                   ).grid(row=0, column=2)

        ttk.Radiobutton(src, text="Video file", variable=self.mode,
                        value="video", command=self.update_state
                        ).grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.video = tk.StringVar()
        self.video_entry = ttk.Entry(src, textvariable=self.video, width=40)
        self.video_entry.grid(row=1, column=1, sticky="ew", padx=4, pady=(6, 0))
        self.browse_btn = ttk.Button(src, text="Browse...",
                                     command=self.pick_video)
        self.browse_btn.grid(row=1, column=2, pady=(6, 0))
        self.no_preview = tk.BooleanVar(value=False)
        self.no_preview_chk = ttk.Checkbutton(
            src, text="Don't show preview (faster)", variable=self.no_preview)
        self.no_preview_chk.grid(row=2, column=1, sticky="w", padx=4)

        # --- what / how ---
        opts = ttk.LabelFrame(body, text="Blur settings", padding=8)
        opts.grid(row=1, column=0, sticky="ew", **pad)

        ttk.Label(opts, text="Blur:").grid(row=0, column=0, sticky="w")
        self.target = tk.StringVar(value="face")
        ttk.Radiobutton(opts, text="Faces", variable=self.target,
                        value="face").grid(row=0, column=1, sticky="w", padx=(0, 12))
        ttk.Radiobutton(opts, text="Whole people", variable=self.target,
                        value="body").grid(row=0, column=2, sticky="w")

        ttk.Label(opts, text="Effect:").grid(row=1, column=0, sticky="w")
        self.effect = tk.StringVar(value="blur")
        ttk.Radiobutton(opts, text="Smooth blur", variable=self.effect,
                        value="blur", command=self.update_state
                        ).grid(row=1, column=1, sticky="w", padx=(0, 12))
        ttk.Radiobutton(opts, text="Pixelate", variable=self.effect,
                        value="pixelate", command=self.update_state
                        ).grid(row=1, column=2, sticky="w")
        self.blocks = tk.IntVar(value=10)
        self.blocks_spin = ttk.Spinbox(opts, from_=2, to=40, width=4,
                                       textvariable=self.blocks)
        self.blocks_spin.grid(row=1, column=3, sticky="w", padx=(8, 4))
        ttk.Label(opts, text="blocks").grid(row=1, column=4, sticky="w")

        ttk.Label(opts, text="Confidence:").grid(row=2, column=0, sticky="w")
        self.conf = tk.IntVar(value=40)
        ttk.Scale(opts, from_=10, to=95, variable=self.conf,
                  command=lambda v: self.conf.set(round(float(v)))
                  ).grid(row=2, column=1, columnspan=3, sticky="ew")
        ttk.Label(opts, textvariable=self.conf).grid(row=2, column=4, sticky="w")

        self.mirror = tk.BooleanVar(value=False)
        self.fullscreen = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Mirror", variable=self.mirror
                        ).grid(row=3, column=1, sticky="w", pady=(4, 0))
        ttk.Checkbutton(opts, text="Start fullscreen", variable=self.fullscreen
                        ).grid(row=3, column=2, columnspan=2, sticky="w",
                               pady=(4, 0))

        # --- buttons + log ---
        btns = ttk.Frame(body)
        btns.grid(row=2, column=0, sticky="ew", **pad)
        self.start_btn = ttk.Button(btns, text="Start", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop,
                                   state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        ttk.Label(btns, text="In the video window: f = fullscreen, q = quit",
                  foreground="gray").pack(side="right")

        self.log = tk.Text(body, height=7, width=64, state="disabled",
                           wrap="word", font=("monospace", 9))
        self.log.grid(row=3, column=0, sticky="ew", **pad)

        self.refresh_cameras()
        self.update_state()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(100, self.poll_output)

    # ----- helpers -----
    def refresh_cameras(self):
        self.cams = list_cameras()
        self.cam_box["values"] = [label for _, label in self.cams] or \
            ["No camera found"]
        # default to the last camera (usually the external one)
        self.cam_box.current(len(self.cams) - 1 if self.cams else 0)

    def pick_video(self):
        path = filedialog.askopenfilename(
            title="Choose a video",
            filetypes=[("Videos", "*.mp4 *.mov *.mkv *.avi *.webm *.m4v"),
                       ("All files", "*")])
        if path:
            self.video.set(path)
            self.mode.set("video")
            self.update_state()

    def update_state(self):
        video = self.mode.get() == "video"
        self.cam_box.configure(state="disabled" if video else "readonly")
        for w in (self.video_entry, self.browse_btn, self.no_preview_chk):
            w.configure(state="normal" if video else "disabled")
        self.blocks_spin.configure(
            state="normal" if self.effect.get() == "pixelate" else "disabled")

    def write(self, text):
        self.log.configure(state="normal")
        # progress lines use \r - keep only the latest one
        if text.startswith("\r"):
            self.log.delete("end-1l linestart", "end-1c")
            text = text.lstrip("\r")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def build_command(self):
        cmd = [sys.executable, "-u", SCRIPT,
               "--target", self.target.get(),
               "--confidence", str(self.conf.get())]
        if self.mode.get() == "video":
            path = self.video.get().strip()
            if not path or not os.path.isfile(path):
                messagebox.showerror("Face Blur", "Choose a video file first.")
                return None
            cmd += ["--video", path]
            if self.no_preview.get():
                cmd.append("--no-preview")
        else:
            if not self.cams:
                messagebox.showerror("Face Blur", "No camera found.")
                return None
            cmd += ["--camera", str(self.cams[self.cam_box.current()][0])]
        if self.effect.get() == "pixelate":
            cmd += ["--pixelate", str(self.blocks.get())]
        if self.mirror.get():
            cmd.append("--mirror")
        if self.fullscreen.get():
            cmd.append("--fullscreen")
        return cmd

    # ----- run / stop -----
    def start(self):
        cmd = self.build_command()
        if not cmd:
            return
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.write("Starting...\n")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True,
                                     bufsize=1)
        threading.Thread(target=self.read_output, args=(self.proc,),
                         daemon=True).start()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

    def read_output(self, proc):
        buf = ""
        while True:
            ch = proc.stdout.read(1)
            if not ch:
                break
            if ch == "\r" and buf:
                self.lines.put(buf)
                buf = ""
            buf += ch
            if ch == "\n":
                self.lines.put(buf)
                buf = ""
        if buf:
            self.lines.put(buf)
        proc.wait()
        self.lines.put(None)

    def poll_output(self):
        try:
            while True:
                item = self.lines.get_nowait()
                if item is None:
                    self.finished()
                else:
                    self.write(item)
        except queue.Empty:
            pass
        self.after(100, self.poll_output)

    def finished(self):
        if self.proc is not None:
            code = self.proc.returncode
            if code not in (0, None, -signal.SIGINT, -signal.SIGTERM):
                self.write(f"\nStopped with an error (code {code}).\n")
            else:
                self.write("Done.\n")
        self.proc = None
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    def stop(self):
        # SIGINT lets a video run finish saving what it has so far
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)

    def on_close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
