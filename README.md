# MultiFaces Comparator

Infos

    Author : Jordan Bertieaux
    Version: 1.0


This project permit the measure of perform Face Detector 

Face Detector used :
    
    - Haarcascade (from OpenCV)
        - haarcascade_frontalface_default.xml
    - Dnn (from OpenCV)
        - Model Coffee 
        - Model TensorFlow
    - HoG (from dlib)
        - dlib model
    - MMOD (from dlib)
        - Model : mmod_human_face_detector.dat
    - TinyFace (Custom Method)
        - Model : weight converted (hr_res101.mat)

Do not use :
    
    Haarcascade from haarcascade because a lot of false positives
    MMOD from dlib because it's very slow

How to Implement ?

    if __name__ == "__main__":
    
        # => Get the Face Detector Class
        fd = FaceDetector() 
        
        # => Get list of Images
        imgPath = list(paths.list_images("IMAGE_TO_DETECT")) 
        cpt = 0
        
        # => Check list content min 1 pictures
        if len(imgPath) > 1:
    
            for img in imgPath:
                img_read = cv2.imread(img)
                vframe = []
    
                cpt += 1
                print("[INFOS] Processing " + str(cpt) + "/" + str(len(imgPath)))
                
                # => Use your Prefered Face Detector
                vframe.append(cv2.resize(fd.detectFaceDlibHog(img_read), (640, 480), interpolation=cv2.INTER_LINEAR))
                
                vframe.append(cv2.resize(fd.detectFaceOpenCVDnn(img_read), (640, 480), interpolation=cv2.INTER_LINEAR))
                
                vframe.append(cv2.resize(fd.detectTinyFace(img_read), (640, 480), interpolation=cv2.INTER_LINEAR))
                
                vframe.append(cv2.resize(fd.detectFaceDlibMMOD(img_read), (640, 480), interpolation=cv2.INTER_LINEAR))
                
                # => show all result of face detectors
                Show_img(vframe)
                vframe.clear()

RESULT:

![alt text](https://github.com/SH4RKNANDO/MultiFaces/blob/master/IMG_RESULT/Result_2.jpg "Logo Title Text 1")


![alt text](https://github.com/SH4RKNANDO/MultiFaces/blob/master/IMG_RESULT/Result_51.jpg "Logo Title Text 1")


![alt text](https://github.com/SH4RKNANDO/MultiFaces/blob/master/IMG_RESULT/Result_60.jpg "Logo Title Text 1")


![alt text](https://github.com/SH4RKNANDO/MultiFaces/blob/master/IMG_RESULT/Result_62.jpg "Logo Title Text 1")


TODO:

    Optimizing TinyFace

Thanks

    PyImageSearch : https://www.pyimagesearch.com/
    Cyndonia : https://github.com/cydonia999/Tiny_Faces_in_Tensorflow

