# MultiFaces Comparator

## Face detector
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

### Face detector do not use :
    
    Haarcascade from haarcascade because a lot of false positives
    MMOD from dlib because it's very slow

### How to use FaceDetector class ?

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

## TINY RESULT :

![alt text](https://github.com/SH4RKNANDO/MultiFaces/blob/master/IMG_RESULT/Result_2.jpg "Logo Title Text 1")


![alt text](https://github.com/SH4RKNANDO/MultiFaces/blob/master/IMG_RESULT/Result_26.jpg "Logo Title Text 1")


![alt text](https://github.com/SH4RKNANDO/MultiFaces/blob/master/IMG_RESULT/Result_28.jpg "Logo Title Text 1")


![alt text](https://github.com/SH4RKNANDO/MultiFaces/blob/master/IMG_RESULT/Result_29.jpg "Logo Title Text 1")


## TINY WAS NOT PERFECT !

![alt text](https://github.com/SH4RKNANDO/MultiFaces/blob/master/IMG_RESULT/Result_6.jpg "Logo Title Text 1")


### TODO:

    Optimizing TinyFace


### Developper Info

    Author : Jordan Bertieaux
    Version: 1.0

###Thanks

    PyImageSearch : https://www.pyimagesearch.com/
    Cyndonia : https://github.com/cydonia999/Tiny_Faces_in_Tensorflow
