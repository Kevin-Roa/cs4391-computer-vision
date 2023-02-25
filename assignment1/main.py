import os
import sys
import time
from textwrap import dedent
from cv2 import imread, imwrite, IMREAD_COLOR
from filters import filters as imfilter


def main(argv):
    img, imgName, filters, output = init(argv)
    setup(img, imgName, filters)

    times = []
    for filter in filters:
        startTime = time.time()

        print(f"\nApplying filter: {filter.desc}")
        out = filter.apply(img)

        endTime = time.time()
        diff = endTime - startTime
        times.append(diff)

        saveImage(out, imgName, filter.name, diff, output)

    printStats(times)


# Print runtime statistics
def printStats(times):
    total = sum(times)
    avg = total / len(times)
    minT = min(times)
    maxT = max(times)

    print("\nRuntime statistics:")
    print(f"Total:   {total:>6.2f}s   |   Minimum: {minT:.2f}s")
    print(f"Average: {avg  :>6.2f}s   |   Maximum: {maxT:.2f}s")


# Save image
def saveImage(img, imgName, filterName, time, dir):
    if dir is None:
        dir = "./output/"
    elif dir[-1] not in "/\\":
        dir += "/"

    out = f"{dir}{imgName}_{filterName}.jpg"
    print(f"   Saving image: {out}")
    print(f"   Time elapsed: {time:.2f} seconds")
    imwrite(out, img)


# Create output directory and print information
def setup(img, imgName, filters):
    # Create output directory
    if not os.path.exists("output"):
        print("Creating output directory \n")
        os.makedirs("output")

    # Print information
    print(f"Image:      {imgName}")
    print(f"Dimensions: {img.shape[:2]}")
    print(f"Filters:    {', '.join([f.name for f in filters])}")


# Check arguments
def init(argv):
    # Print help message if -h or --help is in the arguments
    if "-h" in argv or "--help" in argv:
        help = """
        Description:
            A terminal application to apply various filters individually on a given image.

        Usage: 
            python3 main.py -i <input_image> -o <output_dir> -f <filter1>, <filter2>, ...

        Example: 
            python3 main.py -i ../images/lena30.jpg -o ./output/ -f box7x7, gaussian15x15, motionH15x15, laplacian3x3, canny3x3

        Args:
            -h, --help
                - Describes the usage of the program
                - Lists the available filters
                Ex: python3 main.py -h

            -i, --input
                - Specifies the input image
                * Must provide an input image to run the program
                Ex: python3 main.py -i <input_image>

            -o, --output
                - Specifies the output directory
                * If not provided, the program will save the output images in the same directory as the input image
                Ex: python3 main.py -o <output_directory>

            -f, --filter
                - Specifies the filters to apply to the input image
                - Use a comma-separated list to specify multiple filters
                - Run the program with the `-h` flag to see the available filters
                * If not provided, the program will run ALL available filters
                Ex: python3 main.py -f <filter1>, <filter2>, ...
        """
        print(dedent(help))
        print("Available Filters: \n    " + f", ".join(imfilter.keys()))
        print("    * If no filter is specified, all filters will be applied.")

        sys.exit(1)

    # Check if there are any arguments
    if len(argv) == 0:
        sys.exit("ERROR: Missing arguments. Use -h or --help for help.")

    img = None
    imgName = ""
    imgPath = ""
    filters = []
    output = "./"

    # Check if the image file is specified
    if "-i" in argv or "--input" in argv:
        imgPath = argv[argv.index("-i") + 1]
        imgName = os.path.splitext(os.path.basename(imgPath))[0]

        img = imread(imgPath, IMREAD_COLOR)
        if img is None:
            sys.exit("ERROR: Could not read the image.")
    else:
        sys.exit("ERROR: Missing image file. Use -h or --help for help.")

    # Check if the output directory is specified
    if "-o" in argv or "--output" in argv:
        output = argv[argv.index("-o") + 1]
        if not os.path.exists(output):
            print("Output directory does not exist.")
            os.makedirs(output)
            print(f"Created new output directory: {output}")
    else:
        output = os.path.dirname(imgPath)

    # Check if the filters are specified
    if "-f" in argv or "--filter" in argv:
        # Get every filter after the -f argument
        filters = argv[argv.index("-f") + 1:]
        # Remove every filter that is not in the imfilter dictionary
        filters = [imfilter[f]
                   for f in filters if f.strip(",") in imfilter]

    if len(filters) == 0:
        filters = imfilter.values()

    return img, imgName, filters, output


if __name__ == "__main__":
    main(sys.argv[1:])
