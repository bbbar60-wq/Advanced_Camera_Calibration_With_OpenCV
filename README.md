# Advanced Camera Calibration with OpenCV

This project demonstrates an advanced camera calibration process using OpenCV in Python. The goal of camera calibration is to determine the intrinsic and extrinsic parameters of a camera, which are essential for many computer vision applications like 3D reconstruction, augmented reality, and metric measurements.

---

## 📜 Project Description

Camera calibration corrects for distortions introduced by the lens and perspective, allowing for accurate measurements and analysis of visual data. This project implements the classic calibration technique using a checkerboard pattern. By capturing multiple images of the checkerboard from different angles and positions, the algorithm can calculate the camera's internal parameters (focal length, optical center) and distortion coefficients.

The script automates the process of finding checkerboard corners in a set of images, performing the calibration, and then demonstrating the undistortion effect on a test image.

---

## ✨ Key Features

-   **Automated Corner Detection**: Uses `cv2.findChessboardCorners()` to automatically locate the interior corners of a checkerboard pattern in images.
-   **Camera Matrix and Distortion Coefficients**: Calculates the intrinsic camera matrix (`mtx`) and distortion coefficients (`dist`) using `cv2.calibrateCamera()`.
-   **Image Undistortion**: Applies the calculated calibration parameters to correct lens and perspective distortions in an image using `cv2.undistort()`.
-   **Visual Comparison**: Displays a side-by-side comparison of the original (distorted) and the corrected (undistorted) images to visually verify the calibration results.
-   **Re-projection Error Calculation**: Computes the mean re-projection error to provide a quantitative measure of the calibration accuracy.

---

## 🚀 Getting Started

To get this project running on your local machine, you'll need Python, OpenCV, and a set of calibration images.

### Prerequisites

-   Python 3.8+
-   OpenCV (`opencv-python`)
-   NumPy
-   Matplotlib
-   A collection of images of a checkerboard pattern taken from various angles.

### Installation

1.  **Clone the repository**:
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Install the required packages**:
    ```bash
    pip install opencv-python numpy matplotlib
    ```

3.  **Prepare your calibration images**:
    -   Create a directory named `camera_cal/` in the root of your project.
    -   Place your checkerboard images inside this directory. The script is configured to read `.jpg` files from this folder.
    -   Make sure you have a test image to see the undistortion effect. The script looks for a file named `test_image.jpg`.

---

## 🛠️ Usage

1.  **Ensure your images are in the correct directory** (`camera_cal/`).

2.  **Run the script**:
    ```bash
    python main.py
    ```

3.  **Review the output**:
    -   The script will print the calculated camera matrix and distortion coefficients to the console.
    -   It will also display a window showing the original test image next to its undistorted version.

---

## 💻 Code Structure

The project is organized into a single Python script (`main.py`) for clarity. The main steps are:

1.  **Initialization**:
    -   Set up object points (the 3D coordinates of the checkerboard corners in a perfect, undistorted world).
    -   Initialize arrays to store object points and image points (the 2D coordinates of the detected corners in the images).

2.  **Corner Detection Loop**:
    -   Iterate through all images in the `camera_cal/` directory.
    -   For each image, convert it to grayscale and find the checkerboard corners.
    -   If corners are found, append the object points and image points to their respective lists.

3.  **Calibration**:
    -   Call `cv2.calibrateCamera()` with the collected points to get the camera matrix (`mtx`), distortion coefficients (`dist`), rotation vectors (`rvecs`), and translation vectors (`tvecs`).

4.  **Undistortion and Visualization**:
    -   Load a test image.
    -   Use `cv2.undistort()` with the calibration parameters to correct the image.
    -   Display the original and undistorted images using Matplotlib for comparison.

---

## 🤝 Contributing

Feedback and contributions are welcome! Please feel free to fork the repository, make changes, and submit a pull request.

---

## 📄 License

This project is open-source and available under the MIT License.
