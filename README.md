# RBE/CS 549 Computer Vision: AutoCalib Project

## Project Overview

This project focuses on Camera Calibration using the method presented by Zhengyou Zhang, estimating camera intrinsic and extrinsic parameters and modeling image distortion with a radial-tangential model.

## Key Components

1. **Camera Calibration:** Utilizes Zhang's method to estimate intrinsic and extrinsic camera parameters, providing a robust and flexible approach compared to traditional methods.
2. **Distortion Modeling:** Implements a radial-tangential distortion model to correct image distortion, enhancing the accuracy of the camera calibration process.
3. **Error Estimation and Optimization:** Employs error estimation and optimization techniques to minimize the difference between observed points and points after distortion correction, using a maximum likelihood estimation approach.

## Implementation Details

- Dataset: Uses a checkerboard pattern to capture images from different views for calibration.
- Camera Intrinsic Estimation: Calculates the camera calibration matrix and utilizes the Homography matrix to estimate intrinsic parameters.
- Camera Extrinsic Estimation: Derives the transformation matrix relating world points to camera points from the Homography Matrix and intrinsic parameters.
- Distortion Correction: Applies a radial-tangential model to correct image distortion, enhancing image quality.
- Optimization: Utilizes the Levenberg-Marquardt algorithm for non-linear minimization to refine calibration parameters.

## Results

The project successfully calibrates the camera, correcting distortion and optimizing intrinsic and extrinsic parameters, evidenced by reduced RMS projection error and improved image quality.
![Corners_7](https://github.com/shreyas-chigurupati07/Autocalib/assets/84034817/d749dd1a-fb42-4e43-b70d-b1aa62631ee6)
![Corners_13](https://github.com/shreyas-chigurupati07/Autocalib/assets/84034817/8cad2d08-3cb6-43a3-af60-55380cb945c7)


## How to Run

1. Clone the repository: `git clone [repository-link]`
2. Navigate to the project directory: `cd [project-directory]`
3. Execute the calibration script: `python calibrate.py`

## Dependencies

- Python
- NumPy
- OpenCV
- SciPy



## References

- Zhengyou Zhang's "A flexible new technique for camera calibration."
