# Face-Blur-Coral
A face blurring software using coral ai accelerator.

**Requirments**

Coral AI USB Accelerator

You can get one here: https://gweb-coral-full.uc.r.appspot.com/products/accelerator/

Webcam

**Windows installer**

Work in progress...


**Linux installer (Debian):**

Currently a .deb package that install an app.


**How it works:**

A simple python script calls a AI model run on the Coral that detects objects to blur (People). Then the python script pastes over that selected area a blur patter set by you. Model is pulled from https://github.com/google-coral/examples-camera.
