
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import os
import math
from matplotlib import pyplot as plt
from IPython.display import clear_output
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def plot_show(image,title=""):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.axis("off")
    plt.title(title)
    plt.imshow(image,cmap="Greys_r")
    plt.show()
    
def cut_faces(image, faces_coord):
    faces = []
      
    for (x, y, w, h) in faces_coord:
        w_rm = int(0.2 * w / 2)
        faces.append(image[y: y + h, x + w_rm: x + w - w_rm])
         
    return faces

def get_frame():
    webcam = cv2.VideoCapture(1)
    ret, frame = webcam.read()
    webcam.release()
    
    return frame

def detect_face(frame):
    detector = cv2.CascadeClassifier("xml/frontal_face.xml")

    scale_factor = 1.2
    min_neighbors =5
    min_size = (30,30)
    flags = cv2.CASCADE_SCALE_IMAGE

    faces = detector.detectMultiScale(frame,scaleFactor=scale_factor,
                                     minNeighbors=min_neighbors,
                                     minSize=min_size,
                                     flags=flags)
    
    return faces

def gray_scale(images):
    gray_images = []
    for image in images:
        is_color = len(image.shape) == 3 
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_images.append(image)
    return gray_images

def resize(images,size=(90,90)):
    image_resize = []
    
    for image in images:
        if image.shape < size:
            img_size = cv2.resize(image,size,
                                     interpolation=cv2.INTER_AREA)
        else:
            img_size = cv2.resize(image,size,
                                     interpolation=cv2.INTER_CUBIC)
        image_resize.append(img_size)
        
    return image_resize

def normalize_faces(frame, faces_coord):
    faces = cut_faces(frame, faces_coord)
    #faces = normalize_intensity(faces)
    faces = gray_scale(faces)
    faces = resize(faces)
    return faces

def draw_rectangle(image, coords):
    for (x, y, w, h) in coords:
        w_rm = int(0.2 * w / 2) 
        cv2.rectangle(image, (x + w_rm, y), (x + w - w_rm, y + h), 
                              (0,0,255),2)


# In[3]:


# %load recg_model.py
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output

get_ipython().run_line_magic('matplotlib', 'inline')

# Open a new thread to manage the external cv2 interaction
cv2.startWindowThread()

def plt_show(image,title=""):
    if len(image.shape)==3:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.axis('off')
    plt.title(title)
    plt.imshow(image,cmap="Greys_r")
    plt.show()

class FaceDetector():
    def __init__(self,xml_path):
        self.classifier = cv2.CascadeClassifier(xml_path)

    def detect(self,image):
        scale_factor = 1.2
        min_neighbors=5
        min_size=(30,30)
        flags = cv2.CASCADE_SCALE_IMAGE

        faces_coord = self.classifier.detectMultiScale(image,
                                                       scaleFactor=scale_factor,
                                                       minNeighbors=min_neighbors,
                                                       minSize = min_size,
                                                       flags=flags)

        return faces_coord


class VideoCamera():
    def __init__(self,index=1):
        self.video = cv2.VideoCapture(index)
        self.index=index
        print self.video.isOpened()

    def __del__(self):
        self.video.release()

    def get_frame(self,grayscale=False):
        ret,frame = self.video.read()

        if grayscale:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        return frame


# In[4]:


def collect_dataset():
    images = []
    labels = []
    labels_dic = {}
    #people = [person for person in os.listdir("Male_female/")]
    people = [person for person in os.listdir("trainingData/")]
    #people = [person for person in os.listdir("people/")]
    for i, person in enumerate(people):
        labels_dic[i] = person
        for image in os.listdir("trainingData/" + person):
            if image.endswith('.jpg'):
                images.append(cv2.imread("trainingData/" + person + '/' + image, 
                                     0))
                labels.append(i)
    return (images, np.array(labels), labels_dic)


# In[5]:


images, labels, labels_dic = collect_dataset()


# In[6]:


len(images)
#labels
#labels_dic
#images[0]
#plt.imshow(images[0],cmap=plt.cm.gray)
#plt.show()


# In[ ]:


X_train=np.asarray(images)


# In[ ]:


X_train.shape


# In[ ]:


train=X_train.reshape(-1,8100)


# In[ ]:


train.shape


# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


pca1 = PCA()
pca1.fit(train)
plt.plot(np.cumsum(pca1.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')


# In[ ]:


pca1 = PCA(n_components=400)
new_train=pca1.fit_transform(train)


# In[ ]:


projected = pca1.inverse_transform(new_train)


# In[ ]:


print 'PCA data with 400 components:'
plt.figure(figsize=(10,2.5))
for i in range(1,11):
    plt.subplot(1,10,i)
    plt.axis('off')
    plt.imshow(projected[i].reshape(90,90),cmap=plt.cm.gray)


# In[ ]:


#from skimage.feature import hog
#a=[]

#for img in X_train:
 #   im = hog(img,orientations=9,pixels_per_cell=(8,8),
  #           cells_per_block=(2,2),transform_sqrt=True)
   # a.append(im)
#X_train_new=np.asarray(a)
#X_train_new.shape


# In[ ]:


from sklearn.svm import SVC
from sklearn.svm import LinearSVC


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(new_train,labels,
                                                 test_size=50,
                                                 random_state=0,
                                                 shuffle=True)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


#clf1 = SVC(kernel='rbf',probability=True)
clf2 = LinearSVC()


# In[ ]:


#clf1.fit(X_train_new,labels)
clf2.fit(X_train,y_train)


# In[ ]:


print 'training score:',clf2.score(X_train,y_train)
print 'testing score:',clf2.score(X_test,y_test)


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


y_pred=clf2.predict(X_test)


# In[ ]:


confusion_matrix(y_test,y_pred)


# In[ ]:


print 'misclassify samples:',np.where(y_test!=y_pred)


# In[ ]:


np.bincount(y_test)


# In[ ]:


detector = FaceDetector("xml/frontal_face.xml")
webcam = VideoCamera('videos/b1.mp4')


# ### LinearSVC

# In[ ]:


cv2.startWindowThread()
font=cv2.FONT_HERSHEY_PLAIN
cv2.namedWindow("opencv_face", cv2.WINDOW_AUTOSIZE)



while True:
    frame = webcam.get_frame()
    
    faces_coord = detector.detect(frame) # detect more than one face
    if len(faces_coord):
        faces = normalize_faces(frame, faces_coord) 
        for i, face in enumerate(faces): # for each detected face
            
            
            transform = hog(face,orientations=9,pixels_per_cell=(8,8),
                            cells_per_block=(2,2),transform_sqrt=True)
            
            transform = transform.reshape(1,-1)
            #prob=clf.predict_proba(transform)
            confidence = clf2.decision_function(transform)
            print confidence
           
            
            
            pred = clf2.predict(transform)
            print pred
           
            name=labels_dic[pred[0]].capitalize()
             
            
            #pred = labels_dic[pred[0]].capitalize()
            #threshold = .50
            
            if confidence > 30.0: # apply threshold
                
                cv2.putText(frame, labels_dic[pred[0]].capitalize(),
                            (faces_coord[i][0], faces_coord[i][1] - 10),
                            cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)
            else:
                
                cv2.putText(frame, labels_dic[pred[0]].capitalize(),
                            (faces_coord[i][0], faces_coord[i][1]),
                            cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)
           
                
                
            #cv2.putText(frame,name,(x,y-10),font,2,(0,0,255),2,cv2.LINE_AA)
            
               
           #cv2.putText(frame,'Unknown',(x,y-10),font,2,(0,0,255),2,cv2.LINE_AA)
        clear_output(wait = True)
        draw_rectangle(frame, faces_coord) # rectangle around face
        
    cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 5),
                cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2,
                cv2.LINE_AA)
    
    cv2.imshow("opencv_face", frame) # live feed in external
    if cv2.waitKey(40) & 0xFF == 27:
        
        cv2.destroyAllWindows()
        break


# In[ ]:


del webcam


# ### SVC

# In[ ]:


from sklearn.model_selection import GridSearchCV

#grid search
# cross validation:


# In[ ]:


svc = SVC(kernel='linear',C=.0001)
svc.fit(X_train,y_train)


# In[ ]:


y_pred = svc.predict(X_test)


# In[ ]:


print 'training',svc.score(X_train,y_train)
print 'testing',svc.score(X_test,y_test)


# In[ ]:


svc1 = SVC(kernel='linear',probability=True,C=.0001)
svc1.fit(new_train,labels)


# In[ ]:


print 'training',svc1.score(new_train,labels)


# In[ ]:


detector = FaceDetector("xml/frontal_face.xml")
webcam = VideoCamera(0)


# In[ ]:


cv2.startWindowThread()
font=cv2.FONT_HERSHEY_PLAIN
cv2.namedWindow("opencv_face", cv2.WINDOW_AUTOSIZE)



while True:
    frame = webcam.get_frame()
    
    
    faces_coord = detector.detect(frame) # detect more than one face
    if len(faces_coord):
        faces = normalize_faces(frame, faces_coord) 
        for i, face in enumerate(faces): # for each detected face
            
            
            #cv2.imwrite('trainingData/female/picture_BGR5.jpg',face)
            test = pca1.transform(face.reshape(1,-1))    
            #print test
            #transform = test.reshape(1,-1)
            #print transform
            prob=svc1.predict_proba(test)
            confidence = svc1.decision_function(test)
            print confidence
            print prob
           
            
            
            pred = svc1.predict(test)
            print pred,pred[0]
           
            name=labels_dic[pred[0]].capitalize()
            print name
            
            #pred = labels_dic[pred[0]].capitalize()
            #threshold = .50
            if confidence > .90:
             
                
                cv2.putText(frame, name,
                            (faces_coord[i][0], faces_coord[i][1] - 10),
                            cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)
                
            else:
                
                
                cv2.putText(frame, name,
                            (faces_coord[i][0], faces_coord[i][1]),
                            cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)
           
                
                
            #cv2.putText(frame,name,(x,y-10),font,2,(0,0,255),2,cv2.LINE_AA)
            
               
           #cv2.putText(frame,'Unknown',(x,y-10),font,2,(0,0,255),2,cv2.LINE_AA)
        clear_output(wait = True)
        draw_rectangle(frame, faces_coord) # rectangle around face
        
    cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 5),
                cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2,
                cv2.LINE_AA)
    
    cv2.imshow("opencv_face", frame) # live feed in external
    if cv2.waitKey(5) & 0xFF == 27:
        
        cv2.destroyAllWindows()
        break


# In[ ]:


del webcam

