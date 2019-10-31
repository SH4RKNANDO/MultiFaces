from __future__ import division
import cv2
import dlib

class FaceDetector:
    def __init__(self):
        # OpenCV HAAR
        self._faceCascade = cv2.CascadeClassifier('Data/Model/haarcascade_frontalface_default.xml')

        # OpenCV DNN supports 2 networks.
        # 1. FP16 version of the original caffe implementation ( 5.4 MB )
        # 2. 8 bit Quantized version using Tensorflow ( 2.7 MB )
        DNN = "TF"

        if DNN == "CAFFE":
            self._modelFile = "Data/Model/res10_300x300_ssd_iter_140000_fp16.caffemodel"
            self._configFile = "Data/Model/deploy.prototxt"
            self._net = cv2.dnn.readNetFromCaffe(self._configFile, self._modelFile)
        else:
            self._modelFile = "Data/Model/opencv_face_detector_uint8.pb"
            self._configFile = "Data/Model/opencv_face_detector.pbtxt"
            self._net = cv2.dnn.readNetFromTensorflow(self._modelFile, self._configFile)

        self._conf_threshold = 0.8

        # DLIB HoG
        self._hogFaceDetector = dlib.get_frontal_face_detector()

        # DLIB MMOD
        self._dnnFaceDetector = dlib.cnn_face_detection_model_v1("Data/Model/mmod_human_face_detector.dat")

    def detectFaceOpenCVHaar(self, frame, inHeight=300, inWidth=0):
        frameOpenCVHaar = frame.copy()
        frameHeight = frameOpenCVHaar.shape[0]
        frameWidth = frameOpenCVHaar.shape[1]
        if not inWidth:
            inWidth = int((frameWidth / frameHeight) * inHeight)

        scaleHeight = frameHeight / inHeight
        scaleWidth = frameWidth / inWidth

        frameOpenCVHaarSmall = cv2.resize(frameOpenCVHaar, (inWidth, inHeight))
        frameGray = cv2.cvtColor(frameOpenCVHaarSmall, cv2.COLOR_BGR2GRAY)

        faces = self._faceCascade.detectMultiScale(frameGray)
        bboxes = []
        cv2.putText(frameOpenCVHaar, "OpenCV HaarCascade", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3,
                    cv2.LINE_AA)

        for (x, y, w, h) in faces:
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            cvRect = [int(x1 * scaleWidth), int(y1 * scaleHeight),
                      int(x2 * scaleWidth), int(y2 * scaleHeight)]
            bboxes.append(cvRect)
            cv2.rectangle(frameOpenCVHaar, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (0, 255, 0),
                          int(round(frameHeight / 150)), 4)
        return frameOpenCVHaar

    def detectFaceOpenCVDnn(self, frame):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)

        self._net.setInput(blob)
        detections = self._net.forward()
        bboxes = []
        cv2.putText(frameOpencvDnn, "OpenCV DNN", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self._conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
        return frameOpencvDnn

    def detectFaceDlibHog(self, frame, inHeight=300, inWidth=0):

        frameDlibHog = frame.copy()
        frameHeight = frameDlibHog.shape[0]
        frameWidth = frameDlibHog.shape[1]
        if not inWidth:
            inWidth = int((frameWidth / frameHeight) * inHeight)

        scaleHeight = frameHeight / inHeight
        scaleWidth = frameWidth / inWidth

        frameDlibHogSmall = cv2.resize(frameDlibHog, (inWidth, inHeight))

        frameDlibHogSmall = cv2.cvtColor(frameDlibHogSmall, cv2.COLOR_BGR2RGB)
        faceRects = self._hogFaceDetector(frameDlibHogSmall, 0)
        print(frameWidth, frameHeight, inWidth, inHeight)
        bboxes = []
        cv2.putText(frameDlibHog, "OpenCV HoG", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)
        for faceRect in faceRects:
            cvRect = [int(faceRect.left() * scaleWidth), int(faceRect.top() * scaleHeight),
                      int(faceRect.right() * scaleWidth), int(faceRect.bottom() * scaleHeight)]
            bboxes.append(cvRect)
            cv2.rectangle(frameDlibHog, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (0, 255, 0),
                          int(round(frameHeight / 150)), 4)
        return frameDlibHog

    def detectFaceDlibMMOD(self, frame, inHeight=300, inWidth=0):

        frameDlibMMOD = frame.copy()
        frameHeight = frameDlibMMOD.shape[0]
        frameWidth = frameDlibMMOD.shape[1]
        if not inWidth:
            inWidth = int((frameWidth / frameHeight) * inHeight)

        scaleHeight = frameHeight / inHeight
        scaleWidth = frameWidth / inWidth

        frameDlibMMODSmall = cv2.resize(frameDlibMMOD, (inWidth, inHeight))

        frameDlibMMODSmall = cv2.cvtColor(frameDlibMMODSmall, cv2.COLOR_BGR2RGB)
        faceRects = self._dnnFaceDetector(frameDlibMMODSmall, 0)

        print(frameWidth, frameHeight, inWidth, inHeight)
        bboxes = []

        cv2.putText(frameDlibMMOD, "OpenCV MMOD", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)

        for faceRect in faceRects:
            cvRect = [int(faceRect.rect.left() * scaleWidth), int(faceRect.rect.top() * scaleHeight),
                      int(faceRect.rect.right() * scaleWidth), int(faceRect.rect.bottom() * scaleHeight)]
            bboxes.append(cvRect)
            cv2.rectangle(frameDlibMMOD, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (0, 255, 0),
                          int(round(frameHeight / 150)), 4)
        return frameDlibMMOD