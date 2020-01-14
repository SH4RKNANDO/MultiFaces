from FaceDetector.FaceDetector import FaceDetector
from imutils import paths
import numpy as np
import cv2


# Show Divised Image
def Show_img(frame):
    top = np.hstack([frame[0], frame[1]])
    bottom = np.hstack([frame[2], frame[3]])
    combined = np.vstack([top, bottom])
    cv2.imshow("Face Detection Comparison", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    fd = FaceDetector()
    imgPath = list(paths.list_images("IMAGE_TO_DETECT"))
    cpt = 0

    if len(imgPath) > 1:

        for img in imgPath:
            img_read = cv2.imread(img)
            vframe = []

            cpt += 1
            print("Processing " + str(cpt) + "/" + str(len(imgPath)))

            vframe.append(cv2.resize(fd.detectFaceDlibHog(img_read), (640, 480), interpolation=cv2.INTER_LINEAR))
            vframe.append(cv2.resize(fd.detectFaceOpenCVDnn(img_read), (640, 480), interpolation=cv2.INTER_LINEAR))
            vframe.append(cv2.resize(fd.detectTinyFace(img_read), (640, 480), interpolation=cv2.INTER_LINEAR))
            vframe.append(cv2.resize(fd.detectFaceDlibMMOD(img_read), (640, 480), interpolation=cv2.INTER_LINEAR))
            Show_img(vframe)

            vframe.clear()
