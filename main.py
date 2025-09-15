import cv2
import numpy as np
import os
import argparse
import glob
import json


class CameraCalibrator:
    """
    An advanced class to handle camera calibration and interactive analysis.

    This tool can:
    1. Calibrate a camera using a live video feed (`--mode live`).
    2. Calibrate from a directory of pre-captured images (`--mode dir`).
    3. Display a real-time, side-by-side undistortion view (`--mode undistort`).
    4. Provide an interactive undistortion view with trackbar controls (`--mode undistort_interactive`).
    5. Show a live reprojection error visualization to assess calibration accuracy (`--mode reprojection`).
    """

    def __init__(self, rows=7, cols=7, square_size=25.0, image_dir="calib_images", output_file="calibration_data.npz"):
        self.chessboard_dims = (cols, rows)
        self.square_size = square_size
        self.image_dir = image_dir
        self.output_file = output_file
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objp = np.zeros((self.chessboard_dims[0] * self.chessboard_dims[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chessboard_dims[0], 0:self.chessboard_dims[1]].T.reshape(-1, 2)
        self.objp *= self.square_size
        self.obj_points = []
        self.img_points = []
        self._ensure_dir_exists(self.image_dir)

    def _ensure_dir_exists(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
            print(f"Directory '{path}' was created.")

    def calibrate_from_live_feed(self, camera_index=0):
        # Implementation from previous version...
        print("\nStarting live calibration capture...")
        print("Press 's' to save a frame when a chessboard is detected.")
        print("Press 'q' to finish capturing and start calibration.")
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open camera with index {camera_index}.")
            return
        saved_image_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame from camera.")
                break
            copy_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, self.chessboard_dims, None)
            if found:
                cv2.drawChessboardCorners(frame, self.chessboard_dims, corners, found)
            cv2.putText(frame, f"Saved Images: {saved_image_count}", (30, 40), cv2.FONT_HERSHEY_PLAIN, 1.4, (0, 255, 0),
                        2, cv2.LINE_AA)
            cv2.imshow("Camera Feed - Press 's' to save, 'q' to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and found:
                img_path = os.path.join(self.image_dir, f"image_{saved_image_count}.png")
                cv2.imwrite(img_path, copy_frame)
                print(f"Saved: {img_path}")
                saved_image_count += 1
        cap.release()
        cv2.destroyAllWindows()
        if saved_image_count > 0:
            print(f"\nCaptured {saved_image_count} images. Now calibrating...")
            self.calibrate_from_directory()
        else:
            print("\nNo images were saved. Calibration aborted.")

    def calibrate_from_directory(self):
        images = glob.glob(os.path.join(self.image_dir, '*.png')) + glob.glob(os.path.join(self.image_dir, '*.jpg'))
        if not images:
            print(f"Error: No images found in '{self.image_dir}'.")
            return
        print(f"Found {len(images)} images. Processing...")
        gray_shape = None
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_shape = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_dims, None)
            if ret:
                self.obj_points.append(self.objp)
                refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                self.img_points.append(refined_corners)
            else:
                print(f"Chessboard not found in {fname}. Skipping.")
        if not self.obj_points:
            print("Calibration failed. Could not detect the chessboard in any images.")
            return
        print("\nPerforming camera calibration...")
        try:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_points, self.img_points, gray_shape, None, None)
            if not ret:
                print("Calibration was unsuccessful.")
                return
            print("Calibration successful!")
            print("\nCamera Matrix:\n", mtx)
            print("\nDistortion Coefficients:\n", dist)
            self._calculate_fov(mtx, gray_shape)
            self.save_parameters(mtx, dist, rvecs, tvecs)
        except cv2.error as e:
            print(f"An OpenCV error occurred: {e}")

    def _calculate_fov(self, camera_matrix, image_shape):
        """Calculates and prints the camera's field of view."""
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        width = image_shape[0]
        height = image_shape[1]

        fov_x = 2 * np.arctan(width / (2 * fx)) * (180 / np.pi)
        fov_y = 2 * np.arctan(height / (2 * fy)) * (180 / np.pi)
        fov_diag = 2 * np.arctan(np.sqrt(width ** 2 + height ** 2) / (2 * fx)) * (180 / np.pi)  # Assuming fx approx fy

        print("\n--- Field of View (FOV) ---")
        print(f"Horizontal FOV: {fov_x:.2f} degrees")
        print(f"Vertical FOV:   {fov_y:.2f} degrees")
        print(f"Diagonal FOV:   {fov_diag:.2f} degrees")
        print("---------------------------\n")

    def load_parameters(self):
        """Loads calibration parameters from a .npz or .json file."""
        if not os.path.exists(self.output_file):
            print(f"Error: Calibration file not found at '{self.output_file}'")
            return None, None

        if self.output_file.endswith('.npz'):
            data = np.load(self.output_file)
            cam_matrix = data["camMatrix"]
            dist_coeffs = data["distCoef"]
        elif self.output_file.endswith('.json'):
            with open(self.output_file, 'r') as f:
                data = json.load(f)
            cam_matrix = np.array(data["camera_matrix"])
            dist_coeffs = np.array(data["distortion_coefficients"])
        else:
            print(f"Error: Unsupported file format '{self.output_file}'. Use .npz or .json.")
            return None, None

        return cam_matrix, dist_coeffs

    def save_parameters(self, camera_matrix, dist_coeffs, rvecs, tvecs):
        """Saves the calibration parameters to a .npz or .json file."""
        print(f"\nSaving calibration data to '{self.output_file}'...")
        if self.output_file.endswith('.npz'):
            np.savez(self.output_file, camMatrix=camera_matrix, distCoef=dist_coeffs, rVector=rvecs, tVector=tvecs)
        elif self.output_file.endswith('.json'):
            data = {
                "camera_matrix": camera_matrix.tolist(),
                "distortion_coefficients": dist_coeffs.tolist(),
            }
            with open(self.output_file, 'w') as f:
                json.dump(data, f, indent=4)
        else:
            print(f"Error: Unsupported file format '{self.output_file}'. Use .npz or .json.")
            return
        print("Data saved successfully.")

    def undistort_interactive(self, camera_index=0):
        """Provides an interactive undistortion view with trackbars."""
        cam_matrix, dist_coeffs = self.load_parameters()
        if cam_matrix is None: return

        print("Starting interactive undistortion feed. Press 'q' to quit.")
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}.")
            return

        window_name = "Interactive Undistortion"
        cv2.namedWindow(window_name)
        cv2.createTrackbar("Scaling (alpha)", window_name, 0, 100, lambda x: None)

        while True:
            ret, frame = cap.read()
            if not ret: break

            alpha = cv2.getTrackbarPos("Scaling (alpha)", window_name) / 100.0
            h, w = frame.shape[:2]
            new_cam_matrix, roi = cv2.getOptimalNewCameraMatrix(cam_matrix, dist_coeffs, (w, h), alpha, (w, h))

            undistorted = cv2.undistort(frame, cam_matrix, dist_coeffs, None, new_cam_matrix)

            # Crop the image
            x, y, w, h = roi
            if w > 0 and h > 0:
                undistorted = undistorted[y:y + h, x:x + w]

            cv2.imshow(window_name, undistorted)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

    def live_reprojection_error(self, camera_index=0):
        """Shows a live overlay of reprojected points to visualize accuracy."""
        cam_matrix, dist_coeffs = self.load_parameters()
        if cam_matrix is None: return

        print("Starting live reprojection error visualization. Press 'q' to quit.")
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}.")
            return

        while True:
            ret, frame = cap.read()
            if not ret: break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, self.chessboard_dims, None)

            if found:
                refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                ret, rvec, tvec = cv2.solvePnP(self.objp, refined_corners, cam_matrix, dist_coeffs)

                # Reproject 3D points back to image plane
                reprojected_pts, _ = cv2.projectPoints(self.objp, rvec, tvec, cam_matrix, dist_coeffs)

                # Draw detected corners (green circles)
                for corner in refined_corners:
                    cv2.circle(frame, tuple(corner.ravel().astype(int)), 4, (0, 255, 0), -1)

                # Draw reprojected corners (red crosses)
                for pt in reprojected_pts:
                    cv2.drawMarker(frame, tuple(pt.ravel().astype(int)), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                                   markerSize=5, thickness=1)

            cv2.imshow("Reprojection Error (Green=Detected, Red=Reprojected)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Advanced Camera Calibration and Analysis Tool")
    parser.add_argument(
        '--mode', type=str, required=True,
        choices=['live', 'dir', 'undistort_interactive', 'reprojection'],
        help="'live': Capture & calibrate. 'dir': Calibrate from images. 'undistort_interactive': Live undistortion with controls. 'reprojection': Live reprojection error visualization."
    )
    parser.add_argument('--rows', type=int, default=7, help="Inner corners on chessboard's Y-axis.")
    parser.add_argument('--cols', type=int, default=7, help="Inner corners on chessboard's X-axis.")
    parser.add_argument('--size', type=float, default=25.0, help="Chessboard square size in mm.")
    parser.add_argument('--path', type=str, default="calib_images", help="Directory for calibration images.")
    parser.add_argument('--output', type=str, default="calibration_data.npz",
                        help="Output file for calibration data (.npz or .json).")
    args = parser.parse_args()

    calibrator = CameraCalibrator(
        rows=args.rows, cols=args.cols, square_size=args.size,
        image_dir=args.path, output_file=args.output
    )

    if args.mode == 'live':
        calibrator.calibrate_from_live_feed()
    elif args.mode == 'dir':
        calibrator.calibrate_from_directory()
    elif args.mode == 'undistort_interactive':
        calibrator.undistort_interactive()
    elif args.mode == 'reprojection':
        calibrator.live_reprojection_error()


if __name__ == '__main__':
    main()

