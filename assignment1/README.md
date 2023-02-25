## Assignment 1
A terminal application to apply various filters individually on a given image. 
The program manually implements basic image filters via convolution of the image with specific kernels. `main.py` contains the code to load the image, apply the filters, and save the new images. The filters are implemented in `filters.py`. 

### Filters
- Identity (3x3)
- Box Blur (3x3, 7x7, 15x15)
- Gaussian Blur (3x3, 5x5, 7x7, 15x15)
- Motion Blur (H3x3, V3x3, H15x15, V15x15)
- Laplacian Sharpen (3x3)
- Sobel Edge Detect (3x3)
- Canny Edge Detect

### Usage
1. Clone the repository
   ``` bash
    git clone
   ```
2. Navigate to the assignment1 folder
   ``` bash
    cd assignment1
   ```
3. Run the program
   ``` bash
    python3 main.py -i <input_image> -o <output_dir> -f <filter1>, <filter2>, ...
   ```
4. Example:
   ``` bash
    python3 main.py -i ./images/lena30.jpg -o ./output/ -f box7x7, gaussian15x15, motionH15x15, laplacian3x3, canny3x3
   ```

### Command Line Arguments
- `-h`, `--help`
  - Describes the usage of the program
  - Lists the available filters
  - ```python3 main.py -h```
- `-i`, `--input`
  - Specifies the input image
  - *Must provide an input image to run the program
  - ```python3 main.py -i <input_image>```
- `-o`, `--output`
  - Specifies the output directory
  - *If not provided, the program will save the output images in the same directory as the input image
  - ```python3 main.py -o <output_directory>```
- `-f`, `--filter`
  - Specifies the filters to apply to the input image
  - Use a comma-separated list to specify multiple filters
  - Run the program with the `-h` flag to see the available filters
  - *If not provided, the program will run ALL available filters
  - ```python main.py -f <filter1>, <filter2>, ...```

### TODO
- [ ] Optimize the code (Use numpy functions)
- [ ] Allow combination of filters (ex: gaussian5x5+motionH15x15)