import cv2
import os
from flask import Flask,request,render_template
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import face_recognition
import time
from sklearn.model_selection import GridSearchCV
import threading



#### Defining Flask App
app = Flask(__name__)


#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)


#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name,Roll,Time,ExitTime')


#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


#### extract the face from an image
def extract_faces(img):
    # Apply histogram normalization to the input image
    img = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    face_points = face_detector.detectMultiScale(blur, 1.2, 5)
    for (x, y, w, h) in face_points:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return face_points


#### Identify face using ML model
# def identify_face(facearray):
#     model = joblib.load('static/face_recognition_model.pkl')
#     return model.predict(facearray)


def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    try:
        # Predict the identity of the face
        return model.predict(facearray)
    except ValueError:
        # Handle the error caused by a face not present in the model
        return np.array(['Unknown'])




#### A function which trains the model on all the faces available in faces folder
# def train_model():
#     faces = []
#     labels = []
#     userlist = os.listdir('static/faces')
#     for user in userlist:
#         for imgname in os.listdir(f'static/faces/{user}'):
#             img = cv2.imread(f'static/faces/{user}/{imgname}')
#             resized_face = cv2.resize(img, (100, 100))
#             faces.append(resized_face.ravel())
#             labels.append(user)
#     faces = np.array(faces)
#     knn = KNeighborsClassifier(n_neighbors=5)
#     knn.fit(faces,labels)
#     joblib.dump(knn,'static/face_recognition_model.pkl')

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (100, 100))
            faces.append(resized_face.ravel())
            labels.append(user)
            # Data augmentation - flip the image horizontally and vertically
            flipped_image = cv2.flip(resized_face, 1)
            faces.append(flipped_image.ravel())
            labels.append(user)
            flipped_image = cv2.flip(resized_face, 0)
            faces.append(flipped_image.ravel())
            labels.append(user)

    faces = np.array(faces)

    # Hyperparameter tuning using grid search
    param_grid = {'n_neighbors': [3, 5, 7, 9],
                  'weights': ['uniform', 'distance']}
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(faces, labels)
    best_knn = grid_search.best_estimator_

    # Saving the trained model
    joblib.dump(best_knn, 'static/face_recognition_model.pkl')




#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    timess = df['ExitTime']
    l = len(df)
    return names,rolls,times,timess,l


#### Add Attendance of a specific user check in
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) in list(df['Roll']):
        return
    with open(f'Attendance/Attendance-{datetoday}.csv','a') as f:
        f.write(f'\n{username},{userid},{current_time},')

#### Add Attendance of a specific user check out

def end_attendance(name):
    userid = name.split('_')[1]
    checkout = datetime.now().strftime("%H:%M:%S")

    try:
        df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
        if int(userid) in list(df['Roll']):
            print(f"User ID {userid} found in attendance sheet.")
            print(f"Before update:\n{df}")
            df.loc[(df['Roll'] == int(userid)), 'ExitTime'] = checkout   
            df.to_csv(f'Attendance/Attendance-{datetoday}.csv', index=False)
            print(f"After update:\n{df}")
        else:
            print(f"Error: User ID {userid} not found in attendance sheet.")
    except Exception as e:
        print(f"Error: {e}")


################## ROUTING FUNCTIONS #########################

#### Our main page
@app.route('/')
def home():
    names,rolls,times,timess,l = extract_attendance()    
    return render_template('home.html',names=names,rolls=rolls,times=times,timess=timess,l=l,totalreg=totalreg(),datetoday2=datetoday2) 


#### This function will run when we click on Take Attendance Button
# @app.route('/start',methods=['GET'])
# def start():
#     if 'face_recognition_model.pkl' not in os.listdir('static'):
#         return render_template('home.html',totalreg=totalreg(),datetoday2=datetoday2,mess='There is no trained model in the static folder. Please add a new face to continue.') 

#     cap = cv2.VideoCapture(0)
#     ret = True
#     while ret:
#         ret, frame = cap.read()
#         faces = extract_faces(frame)
#         for (x, y, w, h) in faces: 
#             face = cv2.resize(frame[y:y+h,x:x+w], (50, 50))
#             identified_person = identify_face(face.reshape(1,-1))[0]
#             if identified_person != 'unknown':
#                 cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 0), 2)
#                 add_attendance(identified_person)
#             cv2.putText(frame,f'{identified_person}',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)      
#         cv2.imshow('Attendance',frame)
#         if cv2.waitKey(1)==27:
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#     names,rolls,times,timess,l = extract_attendance()   
#     return render_template('home.html',names=names,rolls=rolls,times=times,l=l,timess=timess,totalreg=totalreg(),datetoday2=datetoday2) 

@app.route('/start', methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2,
                               mess='There is no trained model in the static folder. Please add a new face to continue.')

    cap = cv2.VideoCapture(0)
    ret = True
    model = joblib.load('static/face_recognition_model.pkl')
    recognized_faces = set()  # keep track of recognized faces
    unknown_faces = set()  # keep track of unknown faces
    display_start_time = None
    while ret:
        ret, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            face = cv2.resize(frame[y:y+h, x:x+w], (100, 100))
            # Flatten the resized face region into a 1D numpy array
            flattened_face = face.ravel()
            # Make a prediction using the trained model
            identified_person = model.predict([flattened_face])[0]
            # identified_person = identify_face(face.reshape(1, -1))[0]
            if identified_person != 'unknown':
                if identified_person not in recognized_faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    add_attendance(identified_person)
                    cv2.putText(frame, f'{identified_person}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                            cv2.LINE_AA)
                    display_start_time = time.time()
                    t = threading.Timer(3.0, recognized_faces.add, args=(identified_person,))
                    t.start()
                    # recognized_faces.add(identified_person)
                else:
                    if time.time() - display_start_time > 3: 
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)   
            else:
                if time.time() - display_start_time > 3:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, 'Unknown', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    display_start_time = time.time()
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    recognized_faces.remove(identified_person)
    names, rolls, times, timess, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, timess=timess,
                           totalreg=totalreg(), datetoday2=datetoday2)



####Checkout attendance
@app.route('/end',methods=['GET'])
def end():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html',totalreg=totalreg(),datetoday2=datetoday2,mess='There is no trained model in the static folder. Please add a new face to continue.') 

    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret,frame = cap.read()
        if extract_faces(frame)!=():
            (x,y,w,h) = extract_faces(frame)[0]
            for (x,y,w,h) in extract_faces(frame):
                cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
                face = cv2.resize(frame[y:y+h,x:x+w], (100, 100))
                identified_person = identify_face(face.reshape(1,-1))[0]
                cv2.putText(frame,f'{identified_person}',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
        cv2.imshow('Attendance',frame)
        if cv2.waitKey(1)==27:
            break
    end_attendance(identified_person)
    cap.release()
    cv2.destroyAllWindows()
    names,rolls,times,timess,l = extract_attendance() 
    return render_template('home.html',names=names,rolls=rolls,times=times,l=l,timess=timess,totalreg=totalreg(),datetoday2=datetoday2)     

#### This function will run when we add a new user
@app.route('/add',methods=['GET','POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i,j = 0,0
    while 1:
        _,frame = cap.read()
        faces = extract_faces(frame)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame,f'Images Captured: {i}/50',(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            if j%10==0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
                i+=1
            j+=1
        if j==500:
            break
        cv2.imshow('Adding new User',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names,rolls,times,timess,l = extract_attendance()    
    return render_template('home.html',names=names,rolls=rolls,times=times,timess=timess,l=l,totalreg=totalreg(),datetoday2=datetoday2) 



#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.debug=True
    app.run()