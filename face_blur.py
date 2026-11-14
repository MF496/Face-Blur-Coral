"""Blur faces live from a webcam (or in a video file) using the Coral Edge TPU.

Usage:
  faceblur                     # external camera (/dev/video2)
  faceblur --camera 0          # laptop camera
  faceblur --pixelate          # pixelate instead of blur
  faceblur --pixelate 5        # fewer, bigger blocks (5 across a face)
  faceblur --fullscreen        # start in fullscreen (4:3, black bars)
  faceblur --confidence 60     # only blur faces detected with 60%+ confidence
  faceblur --target body       # blur whole people instead of just faces
  faceblur --video in.mp4      # blur a video file -> in_blurred.mp4
  faceblur --video in.mp4 --output out.mp4 --no-preview
Keys in the video window: f = toggle fullscreen, q or Esc = quit.
"""
import argparse
import os
import re
import shutil
import signal
import subprocess
import sys

import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:  # Python 3.12+: tflite-runtime was renamed ai-edge-litert
    from ai_edge_litert.interpreter import Interpreter, load_delegate

HERE = os.path.dirname(os.path.abspath(__file__))
# folders searched for the models, in order
MODEL_DIRS = [
    os.environ.get("FACEBLUR_MODELS", ""),
    os.path.join(HERE, "models"),
    os.path.expanduser("~/Downloads/examples-camera/all_models"),
]
MODEL_NAMES = {
    # face model: every detection is a face
    "face": ["ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite",
             "mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite"],
    # COCO model: 90 object types, we keep only class 0 = person
    "body": ["ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
             "mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"],
}
PERSON_CLASS_ID = 0
# extra margin added around each box, as a fraction of its size
PADDING = {"face": 0.15, "body": 0.05}


WINDOW = "Blur (f = fullscreen, q = quit)"


def find_model(target):
    for folder in MODEL_DIRS:
        for name in MODEL_NAMES[target]:
            path = os.path.join(folder, name)
            if folder and os.path.exists(path):
                return path
    return None


class Detector:
    """Runs an SSD detection model on the Coral Edge TPU."""

    def __init__(self, model_path):
        try:
            delegate = load_delegate("libedgetpu.so.1")
        except (ValueError, OSError) as e:
            sys.unraisablehook = lambda *_: None  # hide tflite cleanup noise
            raise SystemExit(
                "Could not start the Coral Edge TPU.\n"
                "Is the Coral USB stick plugged in? (Try a different USB port, "
                "or unplug and replug it.)\n"
                f"Details: {e}")
        self.interp = Interpreter(model_path, experimental_delegates=[delegate])
        self.interp.allocate_tensors()
        inp = self.interp.get_input_details()[0]
        self.input_index = inp["index"]
        _, self.in_h, self.in_w, _ = inp["shape"]
        # outputs: boxes, class ids, scores, count (sorted by tensor name)
        outs = sorted(self.interp.get_output_details(), key=lambda d: d["name"])
        self.out_idx = [d["index"] for d in outs]

    def detect(self, frame, threshold):
        """Return a list of (class_id, x0, y0, x1, y1) in frame pixels."""
        fh, fw = frame.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(frame, (int(self.in_w), int(self.in_h))),
                           cv2.COLOR_BGR2RGB)
        self.interp.set_tensor(self.input_index, rgb[np.newaxis])
        self.interp.invoke()
        boxes, classes, scores, count = (self.interp.get_tensor(i)
                                         for i in self.out_idx)
        found = []
        for i in range(int(count[0])):
            if scores[0][i] < threshold:
                continue
            ymin, xmin, ymax, xmax = boxes[0][i]
            found.append((int(classes[0][i]), int(xmin * fw), int(ymin * fh),
                          int(xmax * fw), int(ymax * fh)))
        return found


def detect_screen_size():
    """Return (width, height) of the screen using xrandr, or None."""
    try:
        out = subprocess.run(["xrandr", "--current"], capture_output=True,
                             text=True, timeout=2).stdout
        m = re.search(r"current (\d+) x (\d+)", out)
        if m:
            return int(m.group(1)), int(m.group(2))
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def fit_to_screen(frame, size, stretch=False):
    """Scale frame to the screen. Keeps the aspect ratio with black bars
    unless stretch=True."""
    sw, sh = size
    if stretch:
        return cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
    fh, fw = frame.shape[:2]
    scale = min(sw / fw, sh / fh)
    nw, nh = int(fw * scale), int(fh * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    left, top = (sw - nw) // 2, (sh - nh) // 2
    return cv2.copyMakeBorder(resized, top, sh - nh - top, left, sw - nw - left,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def blur_region(frame, x0, y0, x1, y1, pixelate):
    """pixelate = None for smooth blur, or the number of blocks across the face."""
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return
    if pixelate:
        h, w = roi.shape[:2]
        blocks_x = max(1, min(pixelate, w))
        blocks_y = max(1, min(h, round(blocks_x * h / w)))
        small = cv2.resize(roi, (blocks_x, blocks_y),
                           interpolation=cv2.INTER_LINEAR)
        frame[y0:y1, x0:x1] = cv2.resize(small, (w, h),
                                         interpolation=cv2.INTER_NEAREST)
    else:
        frame[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (99, 99), 30)


def add_audio(source, video_only, output):
    """Copy the audio track from source into the blurred video using ffmpeg.
    Returns True on success. OpenCV can't write audio, so without this step
    the output video is silent."""
    if not shutil.which("ffmpeg"):
        return False
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", video_only, "-i", source,
         "-map", "0:v:0", "-map", "1:a:0?",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", output])
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        prog="faceblur",
        description="Blur faces live from a webcam, or in a video file, "
                    "using the Coral Edge TPU. Run faceblur-gui for the app window.")
    parser.add_argument("--target", choices=["face", "body"], default="face",
                        help="blur just faces, or whole people (default face)")
    parser.add_argument("--model", default=None,
                        help="override the model file (chosen by --target "
                             "if not given)")
    parser.add_argument("--camera", type=int, default=2,
                        help="number N from /dev/videoN (default 2)")
    parser.add_argument("--confidence", "--threshold", dest="confidence",
                        type=float, default=40,
                        help="only blur detections at least this confident, "
                             "in percent (e.g. 60) or as a fraction (0.6); "
                             "default 40")
    parser.add_argument("--stretch", action="store_true",
                        help="in fullscreen, stretch to fill the screen "
                             "instead of keeping the 4:3 shape")
    parser.add_argument("--pixelate", type=int, nargs="?", const=10, default=None,
                        metavar="BLOCKS",
                        help="pixelate instead of blur; BLOCKS = blocks across "
                             "a face (fewer = blockier, default 10)")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--fullscreen", action="store_true",
                        help="start in fullscreen (press f to toggle)")
    parser.add_argument("--screen", metavar="WxH",
                        help="screen size for fullscreen, e.g. 1920x1080 "
                             "(detected automatically if not given)")
    parser.add_argument("--video", metavar="FILE",
                        help="blur a video file (e.g. an .mp4) instead of "
                             "the live camera")
    parser.add_argument("--output", metavar="FILE",
                        help="where to save the blurred video "
                             "(default: <input>_blurred.mp4)")
    parser.add_argument("--no-preview", action="store_true",
                        help="with --video, don't show a window while "
                             "processing (faster)")
    args = parser.parse_args()
    threshold = args.confidence / 100 if args.confidence > 1 else args.confidence

    model = args.model or find_model(args.target)
    if not model or not os.path.exists(model):
        raise SystemExit(f"Model not found: {model or args.target}\n"
                         "Reinstall faceblur, or pass --model PATH")
    detector = Detector(model)

    writer = None
    if args.video:
        if not os.path.exists(args.video):
            raise SystemExit(f"Video not found: {args.video}")
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"Could not open video: {args.video}")
        output = args.output or os.path.splitext(args.video)[0] + "_blurred.mp4"
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # write video-only to a temp file first; audio is added at the end
        temp_output = output + ".tmp.mp4"
        writer = cv2.VideoWriter(temp_output, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (vw, vh))
        if not writer.isOpened():
            raise SystemExit(f"Could not create output video: {output}")
        print(f"Blurring {args.video} -> {output}")
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            raise SystemExit(f"Could not open /dev/video{args.camera}")
    show = not (args.video and args.no_preview)

    if args.screen:
        sw, sh = (int(v) for v in args.screen.lower().split("x"))
        screen_size = (sw, sh)
    else:
        screen_size = detect_screen_size()

    fullscreen = args.fullscreen

    def apply_fullscreen():
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if fullscreen
                              else cv2.WINDOW_NORMAL)

    if show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        apply_fullscreen()

    frame_no = 0
    stopped_early = False
    # Ctrl+C / the app's Stop button: finish cleanly (and still save the video)
    stop = {"now": False}

    def request_stop(*_):
        stop["now"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    while not stop["now"]:
        ok, frame = cap.read()
        if not ok:
            break
        if args.mirror:
            frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]

        objects = detector.detect(frame, threshold)
        if args.target == "body" and not args.model:
            objects = [o for o in objects if o[0] == PERSON_CLASS_ID]

        pad = PADDING[args.target]
        for _, bx0, by0, bx1, by1 in objects:
            pad_x, pad_y = int((bx1 - bx0) * pad), int((by1 - by0) * pad)
            x0, y0 = max(0, bx0 - pad_x), max(0, by0 - pad_y)
            x1, y1 = min(fw, bx1 + pad_x), min(fh, by1 + pad_y)
            blur_region(frame, x0, y0, x1, y1, args.pixelate)

        if writer:
            writer.write(frame)  # save before any fullscreen resizing
            frame_no += 1
            if frame_no % 30 == 0 or frame_no == total:
                pct = f" ({100 * frame_no // total}%)" if total > 0 else ""
                print(f"\r  frame {frame_no}/{total}{pct}", end="", flush=True)

        if not show:
            continue

        if fullscreen:
            size = screen_size
            if not size:
                _, _, ww, wh = cv2.getWindowImageRect(WINDOW)
                size = (ww, wh) if ww > 0 and wh > 0 else None
            if size:
                frame = fit_to_screen(frame, size, args.stretch)

        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or Esc
            stopped_early = True
            break
        if key == ord("f"):
            fullscreen = not fullscreen
            apply_fullscreen()

    if stop["now"]:
        stopped_early = True
    cap.release()
    if show:
        cv2.destroyAllWindows()

    if writer:
        writer.release()
        print()
        if stopped_early:
            print("Stopped early - the saved video only has the part "
                  "processed so far.")
        if add_audio(args.video, temp_output, output):
            os.remove(temp_output)
            print(f"Saved {output} (with audio)")
        else:
            os.replace(temp_output, output)
            print(f"Saved {output} (no audio - install ffmpeg to keep "
                  "the sound)")


if __name__ == "__main__":
    main()
