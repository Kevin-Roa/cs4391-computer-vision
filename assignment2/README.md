## Assignment 1
A terminal application to apply image segmentation techniques on an image.
The code was mainly tailored for the sample image provided, results may vary for other images.

Techniques used:
- Histogram thresholding
  - Binary threshold (Black/White)
  - Multi-level threshold (Grayscale)
- K-means clustering
  - Get segments via color quanatization (K = 5)
- Edge and contour detection
  - Canny edge detection
  - Topological structual analysis
  - Results tailored to the sample image
    - Segment the boy and dog, show their face contours
    - Minimize false contours on grass/trees


### Usage
1. Clone the repository
   ``` bash
    git clone
   ```
2. Navigate to the assignment1 folder
   ``` bash
    cd assignment2
   ```
3. Run the program
   ``` bash
    python3 main.py <input_image>
   ```
4. Example:
   ``` bash
    python3 main.py ../images/sample.jpg
   ```
5. The program will save the output images in an output folder where the program is run.
