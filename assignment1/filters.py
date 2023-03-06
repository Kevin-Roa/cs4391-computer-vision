import numpy as np
import cv2


filters = dict()


class Filter:
    def __init__(self, name, desc, kernel=[]):
        self.name = name
        self.desc = desc
        self.kernel = kernel
        self._offset = self.kernel.shape[0] // 2

    # Apply the filter to the image
    def apply(self, img):
        out = self.__pad(img)
        out = self._process(out)
        out = self.__unpad(out)
        return out

    # Steps to apply the filter
    def _process(self, img):
        return self.__iterateImg(img)

    # Convert image to grayscale
    def _toGray(self, img):
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    # Iterate over the image and apply the filter
    def __iterateImg(self, img):
        out = np.zeros(img.shape, np.float32)
        for r in range(self._offset, img.shape[0] - self._offset):
            for c in range(self._offset, img.shape[1] - self._offset):
                self.__applyFilter(img, out, r, c)
        return out

    # Apply the filter to the 3 channel image
    def __applyFilter(self, img, out, r, c):
        if len(img.shape) == 2:
            out.itemset((r, c), self.__applyKernel(img, r, c, 0))
        else:
            for chnl in range(img.shape[2]):
                out.itemset((r, c, chnl), self.__applyKernel(img, r, c, chnl))

    # Get the new pixel value
    def __applyKernel(self, img, r, c, chnl):
        ret = 0
        for i in range(self.kernel.shape[0]):
            for j in range(self.kernel.shape[1]):
                if len(img.shape) == 2:
                    ret += img[r - self._offset + i, c -
                               self._offset + j] * self.kernel[i, j]
                else:
                    ret += img[r - self._offset + i, c -
                               self._offset + j, chnl] * self.kernel[i, j]
        return ret

    # Pad the image with a reflected border of the kernel radius
    def __pad(self, img):
        return cv2.copyMakeBorder(img, self._offset, self._offset, self._offset, self._offset, cv2.BORDER_REFLECT)

    # Unpad the image
    def __unpad(self, img):
        return img[self._offset:-self._offset, self._offset:-self._offset,]


# Custom Laplacian filter process
class Laplacian(Filter):
    def __init__(self, name, desc, kernel=[]):
        super().__init__(name, desc, kernel)

    def _process(self, img):
        out = filters["gaussian3x3"].apply(img)
        out = super()._process(out)
        out = img + out
        return out


# Custom Sobel filter process
class Sobel(Filter):
    def __init__(self, name, desc, kernel=[]):
        super().__init__(name, desc, kernel)

    def _process(self, img):
        out = self._toGray(img)

        # Apply horizontal and vertical sobel kernels
        Ix = super()._process(out)
        self.kernel = self.kernel.T
        Iy = super()._process(out)

        # Calculate the magnitude of the gradient
        out = np.hypot(Ix, Iy)
        out = out / out.max() * 255

        # Calculate the direction of the gradient
        theta = np.arctan2(Iy, Ix)

        out = self.__non_max_suppression(out, theta)

        return out

    # Code by Sofiane Sahir
    # Available at: https://gist.github.com/FienSoP/03ed9b0eab196dde7b66f452725a42ac#file-nonmaxsuppression-py
    def __non_max_suppression(self, img, D):
        M, N = img.shape
        Z = np.zeros((M, N), dtype=np.float32)
        angle = D * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1, M-1):
            for j in range(1, N-1):
                try:
                    q = 255
                    r = 255

                # angle 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = img[i, j+1]
                        r = img[i, j-1]
                    # angle 45
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = img[i+1, j-1]
                        r = img[i-1, j+1]
                    # angle 90
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = img[i+1, j]
                        r = img[i-1, j]
                    # angle 135
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = img[i-1, j-1]
                        r = img[i+1, j+1]

                    if (img[i, j] >= q) and (img[i, j] >= r):
                        Z[i, j] = img[i, j]
                    else:
                        Z[i, j] = 0

                except IndexError as e:
                    pass

        return Z


# Custom Canny filter process
class Canny(Filter):
    def __init__(self, name, desc, kernel=[]):
        super().__init__(name, desc, kernel)

    def _process(self, img):
        out = self._toGray(img)
        out = filters["gaussian5x5"].apply(out)
        out = filters["sobel3x3"].apply(out)
        out = super()._process(out)

        hThresh = np.max(out) * .65
        lThresh = hThresh * .35

        out = cv2.inRange(out, lThresh, hThresh)
        return out


# Identity
filters["identity"] = (Filter("identity", "Identity Matrix 3x3", np.array(
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.float32)))

# Box filters
filters["box3x3"] = (
    Filter("box3x3", "Box Blur 3x3", np.ones((3, 3), np.float32) / 9))
filters["box7x7"] = (
    Filter("box7x7", "Box Blur 7x7", np.ones((7, 7), np.float32) / 49))
filters["box15x15"] = (
    Filter("box15x15", "Box Blur 15x15", np.ones((15, 15), np.float32) / 225))


# Motion filters
filters["motionH3x3"] = (Filter("motionH3x3", "Horizontal Motion Blur 3x3", np.array(
    [[0, 0, 0], [1, 1, 1], [0, 0, 0]], np.float32) / 3))
filters["motionV3x3"] = (Filter("motionV3x3", "Vertical Motion Blur 3x3", np.array(
    [[0, 1, 0], [0, 1, 0], [0, 1, 0]], np.float32) / 3))

__m15h = np.zeros((15, 15), np.float32)
__m15v = __m15h.copy()
__m15h[7, :] = 1 / 15
__m15v[:, 7] = 1 / 15
filters["motionH15x15"] = (
    Filter("motionH15x15", "Horizontal Motion Blur 15x15", __m15h))
filters["motionV15x15"] = (
    Filter("motionV15x15", "Vertical Motion Blur 15x15", __m15v))


# Laplacian filters
__l3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32)
filters["laplacian3x3"] = (
    Laplacian("laplacian3x3", "Laplacian Sharpen 3x3", __l3))


# Sobel filters
__s3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
filters["sobel3x3"] = (Sobel("sobel3x3", "Sobel Edge Detect 3x3", __s3))


# Canny filters
__c3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32)
filters["canny3x3"] = (Canny("canny3x3", "Canny Edge Detect 3x3", __c3))


# Gaussian filters
filters["gaussian3x3"] = (Filter("gaussian3x3", "Gaussian Blur 3x3", np.array(
    [[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16))
filters["gaussian5x5"] = (Filter("gaussian5x5", "Gaussian Blur 5x5", np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [
    6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]], np.float32) / 256))

__g7 = np.array([
    [4.4156577513526845, 8.144367401843168, 11.758784719942968, 13.290083146997514,
        11.758784719942968, 8.144367401843168, 4.4156577513526845],
    [8.144367401843168, 15.021707775220126, 21.68824405130984, 24.512615344120363,
        21.68824405130984, 15.021707775220126, 8.144367401843168],
    [11.758784719942968, 21.68824405130984, 31.31334579714814, 35.39115471252564,
        31.31334579714814, 21.68824405130984, 11.758784719942968],
    [13.290083146997514, 24.512615344120363, 35.39115471252564,
     40, 35.39115471252564, 24.512615344120363, 13.290083146997514],
    [11.758784719942968, 21.68824405130984, 31.31334579714814, 35.39115471252564,
        31.31334579714814, 21.68824405130984, 11.758784719942968],
    [8.144367401843168, 15.021707775220126, 21.68824405130984, 24.512615344120363,
        21.68824405130984, 15.021707775220126, 8.144367401843168],
    [4.4156577513526845, 8.144367401843168, 11.758784719942968, 13.290083146997514, 11.758784719942968, 8.144367401843168, 4.4156577513526845]], np.float32)
__g7 /= np.sum(__g7)
filters["gaussian7x7"] = (
    Filter("gaussian7x7", "Gaussian Blur 7x7", __g7))

__g15 = np.array([
    [0.0002450710232716417, 0.001205054269910814, 0.004636497992069249, 0.013960066533692805, 0.032896044800877794, 0.0606744204402545, 0.08760133387435295, 0.09900929719408005,
        0.08760133387435295, 0.0606744204402545, 0.032896044800877794, 0.013960066533692805, 0.004636497992069249, 0.001205054269910814, 0.0002450710232716417],
    [0.001205054269910814, 0.0059254487700926015, 0.022798418304162228, 0.06864392844199707, 0.16175522802031045, 0.2983460404653774, 0.43075007410468996, 0.48684489398955816,
        0.43075007410468996, 0.2983460404653774, 0.16175522802031045, 0.06864392844199707, 0.022798418304162228, 0.0059254487700926015, 0.001205054269910814],
    [0.004636497992069249, 0.022798418304162228, 0.08771789232150198, 0.264110459035691, 0.6223601780012602, 1.14789918769543, 1.6573293863502807, 1.8731566119414789,
        1.6573293863502807, 1.14789918769543, 0.6223601780012602, 0.264110459035691, 0.08771789232150198, 0.022798418304162228, 0.004636497992069249],
    [0.013960066533692805, 0.06864392844199707, 0.264110459035691, 0.7952121594119149, 1.8738689216903963, 3.4562182624926265, 4.990065463474622, 5.639901273646251,
        4.990065463474622, 3.4562182624926265, 1.8738689216903963, 0.7952121594119149, 0.264110459035691, 0.06864392844199707, 0.013960066533692805],
    [0.032896044800877794, 0.16175522802031045, 0.6223601780012602, 1.8738689216903963, 4.4156577513526845, 8.14436740184317, 11.75878471994297, 13.290083146997516,
        11.75878471994297, 8.14436740184317, 4.4156577513526845, 1.8738689216903963, 0.6223601780012602, 0.16175522802031045, 0.032896044800877794],
    [0.0606744204402545, 0.2983460404653774, 1.14789918769543, 3.4562182624926265, 8.14436740184317, 15.021707775220127, 21.688244051309844, 24.512615344120363,
        21.688244051309844, 15.021707775220127, 8.14436740184317, 3.4562182624926265, 1.14789918769543, 0.2983460404653774, 0.0606744204402545],
    [0.08760133387435295, 0.43075007410468996, 1.6573293863502807, 4.990065463474622, 11.75878471994297, 21.688244051309844, 31.313345797148152, 35.39115471252565,
        31.313345797148152, 21.688244051309844, 11.75878471994297, 4.990065463474622, 1.6573293863502807, 0.43075007410468996, 0.08760133387435295],
    [0.09900929719408005, 0.48684489398955816, 1.8731566119414789, 5.639901273646251, 13.290083146997516, 24.512615344120363, 35.39115471252565, 40,
        35.39115471252565, 24.512615344120363, 13.290083146997516, 5.639901273646251, 1.8731566119414789, 0.48684489398955816, 0.09900929719408005],
    [0.08760133387435295, 0.43075007410468996, 1.6573293863502807, 4.990065463474622, 11.75878471994297, 21.688244051309844, 31.313345797148152, 35.39115471252565,
        31.313345797148152, 21.688244051309844, 11.75878471994297, 4.990065463474622, 1.6573293863502807, 0.43075007410468996, 0.08760133387435295],
    [0.0606744204402545, 0.2983460404653774, 1.14789918769543, 3.4562182624926265, 8.14436740184317, 15.021707775220127, 21.688244051309844, 24.512615344120363,
        21.688244051309844, 15.021707775220127, 8.14436740184317, 3.4562182624926265, 1.14789918769543, 0.2983460404653774, 0.0606744204402545],
    [0.032896044800877794, 0.16175522802031045, 0.6223601780012602, 1.8738689216903963, 4.4156577513526845, 8.14436740184317, 11.75878471994297, 13.290083146997516,
        11.75878471994297, 8.14436740184317, 4.4156577513526845, 1.8738689216903963, 0.6223601780012602, 0.16175522802031045, 0.032896044800877794],
    [0.013960066533692805, 0.06864392844199707, 0.264110459035691, 0.7952121594119149, 1.8738689216903963, 3.4562182624926265, 4.990065463474622, 5.639901273646251,
        4.990065463474622, 3.4562182624926265, 1.8738689216903963, 0.7952121594119149, 0.264110459035691, 0.06864392844199707, 0.013960066533692805],
    [0.004636497992069249, 0.022798418304162228, 0.08771789232150198, 0.264110459035691, 0.6223601780012602, 1.14789918769543, 1.6573293863502807, 1.8731566119414789,
        1.6573293863502807, 1.14789918769543, 0.6223601780012602, 0.264110459035691, 0.08771789232150198, 0.022798418304162228, 0.004636497992069249],
    [0.001205054269910814, 0.0059254487700926015, 0.022798418304162228, 0.06864392844199707, 0.16175522802031045, 0.2983460404653774, 0.43075007410468996, 0.48684489398955816,
        0.43075007410468996, 0.2983460404653774, 0.16175522802031045, 0.06864392844199707, 0.022798418304162228, 0.0059254487700926015, 0.001205054269910814],
    [0.0002450710232716417, 0.001205054269910814, 0.004636497992069249, 0.013960066533692805, 0.032896044800877794, 0.0606744204402545, 0.08760133387435295, 0.09900929719408005, 0.08760133387435295, 0.0606744204402545, 0.032896044800877794, 0.013960066533692805, 0.004636497992069249, 0.001205054269910814, 0.0002450710232716417]])
__g15 /= np.sum(__g15)
filters["gaussian15x15"] = (
    Filter("gaussian15x15", "Gaussian Blur 15x15", __g15))
