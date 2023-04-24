import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np


def main(imgPath):
    img = readImg(imgPath)

    hist = getHistogram(img)
    pltHist = plotHistogram(hist)
    savePlt(pltHist, "histogram.png")

    thresh = imgBinaryThreshold(img)
    saveImg(thresh, "binary_threshold.png")

    thresh = imgMultiThreshold(img)
    saveImg(thresh, "multi_threshold.png")

    kmean = imgKMeanCluster(img, 5)
    saveImg(kmean, "kmean_cluster.png")

    edgeContour = imgEdgeDetection(img)
    saveImg(edgeContour, "edge_contour.png")

    return


def readImg(imgPath):
    img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    if img is None:
        print("Error: Image could not be read")
        sys.exit(1)
    return img


def getHistogram(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([grey], [0], None, [256], [0, 256])
    return hist


def plotHistogram(hist):
    fig = plt.figure()
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.close(fig)
    return fig


def imgBinaryThreshold(img):
    copy = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(copy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh


def imgMultiThreshold(img):
    copy = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = np.zeros(copy.shape, dtype=np.uint8)
    thresh[copy < 79] = 0
    thresh[(copy >= 80) & (copy < 149)] = 64
    thresh[(copy >= 150) & (copy < 192)] = 128
    thresh[copy >= 192] = 255

    return thresh


def imgKMeanCluster(img, regions):
    # Color quantization using K-Means
    # https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html
    # Could have used random values but wanted to get best colors from the image
    Z = np.float32(img.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(
        Z, regions, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    center = np.uint8(center)
    res = center[label.flatten()]
    res = res.reshape((img.shape))

    return res


def imgEdgeDetection(img):
    copy = img.copy()

    blur = cv2.GaussianBlur(copy, (7, 7), 0)
    median = np.median(blur)
    lower = int(max(0, (1.0 - 0.7) * median))
    upper = int(min(255, (1.0 + 0.7) * median))

    edges = cv2.Canny(blur, lower, upper)
    contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    cv2.drawContours(copy, contours[0], -1, (255, 255, 0), 2)

    return copy


def saveImg(img, imgName):
    out = os.path.join(".\\output", imgName)
    cv2.imwrite(out, img)
    print(f"Saved {imgName} to {out}")


def savePlt(pltFig, imgName):
    out = os.path.join(".\\output", imgName)
    pltFig.savefig(out)
    print(f"Saved {imgName} to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <image>")
        sys.exit(1)

    if not os.path.exists(".\\output"):
        print("Output directory does not exist.")
        os.makedirs(".\\output")
        print(f"Created new output directory")

    main(sys.argv[1])
