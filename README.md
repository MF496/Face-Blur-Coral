# Face-Blur-Coral

Blur faces (or whole people) live from a webcam or in a video file, using a **Google Coral USB Accelerator**.

[**⬇ Download the latest release**](https://github.com/USERNAME/Face-Blur-Coral/releases/latest)

## Requirements

- [Coral USB Accelerator](https://coral.ai/products/accelerator) with its driver installed (`libedgetpu1-std`)
- A webcam (or a video file to blur)
- Linux (Debian / Ubuntu)

## Install (Linux)

Download [faceblur_amd64.deb](https://github.com/USERNAME/Face-Blur-Coral/releases/latest/download/faceblur_amd64.deb) and install it:

```bash
sudo apt install ./faceblur_amd64.deb
```

Or do both in the terminal:

```bash
wget https://github.com/USERNAME/Face-Blur-Coral/releases/latest/download/faceblur_amd64.deb
sudo apt install ./faceblur_amd64.deb
```

Then open **Face Blur** from the app menu, or run `faceblur --help` in a terminal.

**Windows installer:** work in progress.

## How it works

A Python script sends each camera frame to an AI model running on the Coral, which finds the faces (or people) in the picture. The script then covers each of those areas with a blur or pixelation, set by you.

The models come from [google-coral/test_data](https://github.com/google-coral/test_data).

## Uninstall

```bash
sudo apt remove faceblur
```
