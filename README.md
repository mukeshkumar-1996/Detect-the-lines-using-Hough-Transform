#  Lane Detection

##  Aim

To implement a basic lane detection pipeline using OpenCV by completing missing code segments at specified locations.

---

## Learning Objective

* Understand each stage of image processing
* Learn how to build a complete computer vision pipeline
* Practice writing code in guided sections

**Important Instruction:**
👉 Write code **ONLY in places marked as `# Your Code Here`**
👉 Do NOT modify any other part of the code

---

##  Software Used

* Anaconda – Python 3.7
* Jupyter Notebook / VS Code
* OpenCV (cv2)
* NumPy
* Matplotlib

---

##  Algorithm & Explanation

---

###  Step 1: Import Libraries

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

---

###  Step 2: Read the Image

```python
# Read the image using OpenC

###
image = cv2.imread("road.jpg")

if image is None:
    raise FileNotFoundError("Could not load 'road.jpg'. Check the file path.")

# Keep a copy in RGB for correct display with matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

###
```

---

###  Step 3: Convert to Grayscale

```python
# Convert to grayscale.

###

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
###
```

---

###  Step 4: Display Images

```python
plt.figure(figsize=(10,5))

###
plt.figure(figsize=(10,5))


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.tight_layout()
plt.show()

###
```

---

###  Step 5: Thresholding

```python
# Apply thresholding
###
threshold = 

threshold = 150
_, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(6, 6))
plt.imshow(thresh, cmap="gray")
plt.title("Thresholded Image")
plt.axis("off")
plt.show()
###
```

---

###  Step 6: Region of Interest (ROI)

```python
# ROI masking already provided
# (Do not modify)
```
# ROI masking already provided
# (Do not modify)
height, width = thresh.shape

roi_vertices = np.array([[
    (int(0.1 * width), height),
    (int(0.45 * width), int(0.6 * height)),
    (int(0.55 * width), int(0.6 * height)),
    (int(0.9 * width), height)
]], dtype=np.int32)

mask = np.zeros_like(thresh)
cv2.fillPoly(mask, roi_vertices, 255)
roi_masked = cv2.bitwise_and(thresh, mask)

plt.figure(figsize=(6, 6))
plt.imshow(roi_masked, cmap="gray")
plt.title("ROI Masked Image")
plt.axis("off")
plt.show()
---

### Step 7: Edge Detection (Canny)

```python
# Perform Edge Detection

###
# Perform Edge Detection

###
edges = cv2.Canny(roi_masked, 50, 150)

plt.figure(figsize=(6, 6))
plt.imshow(edges, cmap="gray")
plt.title("Edge Detected Image")
plt.axis("off")
plt.show()
###
###
```

---

###  Step 8: Gaussian Blur

```python
# Apply Gaussian Blur

###
# Apply Gaussian Blur

###
smoothed = cv2.GaussianBlur(edges, (5, 5), 0)

plt.figure(figsize=(6, 6))
plt.imshow(smoothed, cmap="gray")
plt.title("Smoothed (Blurred) Edge Image")
plt.axis("off")
plt.show()
###
###
```

---

###  Step 9: Hough Transform

```python
# Detect lines using Hough Transform

###
# Detect lines using Hough Transform

###
lines = cv2.HoughLinesP(
    smoothed,
    rho=2,
    theta=np.pi / 180,
    threshold=50,
    minLineLength=40,
    maxLineGap=100
)

line_image = np.zeros_like(image)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

line_image_rgb = cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6, 6))
plt.imshow(line_image_rgb)
plt.title("Detected Lines")
plt.axis("off")
plt.show()
###
---
###
```

---

### Step 10: Lane Detection Logic

```python
# Already implemented
# (Do not modify)
```
# Already implemented
# (Do not modify)
final_output = cv2.addWeighted(image, 0.8, line_image, 1.0, 0.0)
final_output_rgb = cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6, 6))
plt.imshow(final_output_rgb)
plt.title("Final Lane Detection Output")
plt.axis("off")
plt.show()
---

##  Expected Output

* Original image
* <img width="401" height="608" alt="Screenshot 2026-09-16 160816" src="https://github.com/user-attachments/assets/44432c7f-fde6-466b-9be8-0c0a104c23a3" />

* Grayscale image
* <img width="403" height="619" alt="Screenshot 2026-09-16 160822" src="https://github.com/user-attachments/assets/58eea724-4acb-47cc-8f54-2dfceb8fabf4" />

* Thresholded image
* <img width="423" height="625" alt="Screenshot 2026-09-16 160831" src="https://github.com/user-attachments/assets/4dbc9e0c-7e5a-4552-b941-5ce61e9f0053" />

* ROI masked image
* <img width="462" height="635" alt="Screenshot 2026-09-16 160840" src="https://github.com/user-attachments/assets/03f86c01-9852-4fc4-8f9c-58ffb73d2601" />

* Edge detected image
* <img width="478" height="634" alt="Screenshot 2026-09-16 160848" src="https://github.com/user-attachments/assets/9dba13a8-714a-4652-be49-e43a8818bcba" />

* Smoothed image
* <img width="450" height="625" alt="Screenshot 2026-09-16 160857" src="https://github.com/user-attachments/assets/3631cbeb-4e9c-4d59-ab18-1c093b82682a" />

* Detected lines
* <img width="431" height="627" alt="Screenshot 2026-09-16 160905" src="https://github.com/user-attachments/assets/84315262-4caa-43a8-81fb-cf3aa91fea20" />

* Final lane detection output
* <img width="439" height="629" alt="Screenshot 2026-09-16 160912" src="https://github.com/user-attachments/assets/ff9dde54-c851-4a21-b784-bb9ab6187e6e" />


---

##  Instructions

* Fill ONLY in `# Your Code Here` sections
* Do NOT change existing code
* Run step-by-step
* Verify outputs

---

## Result

Thus, the lane detection pipeline is successfully implemented by completing the missing code sections. The system detects and highlights lane lines effectively.

---

##  Developed By

* **Name:** Mukeshkumar V
* **Register No:** 212225230193
