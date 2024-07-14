#All Library Imports
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer
from tkinter.ttk import *
from tkinter import *
import tkinter.messagebox

# creating tkinter window
root = Tk()
root.title('Driver Drowsiness Detection v1.0')
root.geometry('500x570')
root.resizable(0,0)
root.iconbitmap(r"C:\Users\HP\Documents\Driver Drowsiness\Exe Testing\Drowsiness detection\icon.ico")
frame = Frame(root, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH,expand=1)
frame.config(background='orange')
label = Label(frame, text="Driver Cam",bg='orange',fg='white',font=('Times 35 bold'))
label.pack(side=TOP)
filename = PhotoImage(file=r"C:\Users\HP\Documents\Driver Drowsiness\Exe Testing\Drowsiness detection\Drowsy.png")
background_label = Label(frame,image=filename)
background_label.pack(side=TOP)


    
# Creating Menubar
menubar = Menu(root)

#Menu defined

def About():
    tkinter.messagebox.showinfo("About","Driver Cam version v1.0\n Developed Using\n-Tensorflow\n-Keras\n-Pygame\n-OpenCV\n-Numpy\n-Tkinter\n In Python 3.7")
    
def Contri():
   tkinter.messagebox.showinfo("Contributors","\n1.Saadat Ali\n2. Salman Kazim \n3. Waqar ul Hassan \n")



menu = Menu(root)


subm1 = Menu(menu)
menu.add_cascade(label="File",menu=subm1)
subm1.add_command(label ='Exit', command = root.destroy)

subm2 = Menu(menu)
menu.add_cascade(label="About",menu=subm2)
subm2.add_command(label="Driver Cam",command=About)
subm2.add_command(label="Contributors",command=Contri)


# display Menu
root.config(menu=menu)

#Main Functionality
def webdetRec():
    mixer.init()
    sound = mixer.Sound('alarm.wav')

    face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
    leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
    reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')



    lbl=['Close','Open']

    model = load_model('models/cnn.h5')
    path = os.getcwd()
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    count=0
    score=0
    thicc=2
    rpred=[99]
    lpred=[99]

    while(True):
        ret, frame = cap.read()
        height,width = frame.shape[:2] 

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
        left_eye = leye.detectMultiScale(gray)
        right_eye =  reye.detectMultiScale(gray)

        cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , -1)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,128,0) , 1 )

        for (x,y,w,h) in right_eye:
            #rectangle around rigth eye
            #cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,128,0) , 1 )
            r_eye=frame[y:y+h,x:x+w]
            count=count+1
            r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye,(24,24))
            r_eye= r_eye/255
            r_eye=  r_eye.reshape(24,24,-1)
            r_eye = np.expand_dims(r_eye,axis=0)
            rpred = model.predict_classes(r_eye)
            if(rpred[0]==1):
                lbl='Open' 
            if(rpred[0]==0):
                lbl='Closed'
            break

        for (x,y,w,h) in left_eye:
            #rectangle around left eye
            #cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,128,0) , 1 )
            l_eye=frame[y:y+h,x:x+w]
            count=count+1
            l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
            l_eye = cv2.resize(l_eye,(24,24))
            l_eye= l_eye/255
            l_eye=l_eye.reshape(24,24,-1)
            l_eye = np.expand_dims(l_eye,axis=0)
            lpred = model.predict_classes(l_eye)
            if(lpred[0]==1):
                lbl='Open'   
            if(lpred[0]==0):
                lbl='Closed'
            break

        if(rpred[0]==0 and lpred[0]==0):
            score=score+1
            cv2.putText(frame,"Closed",(10,height-20), font, 1,(0,0,255),1,cv2.LINE_AA)
        # if(rpred[0]==1 or lpred[0]==1):
        else:
            score=score-1
            cv2.putText(frame,"Open",(10,height-20), font, 1,(0,128,0),1,cv2.LINE_AA)


        if(score<0):
            score=0   
        cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        if(score>15):
            #Because the person is asleep, Trigger an alarm.
            cv2.imwrite(os.path.join(path,'image.jpg'),frame)
            try:
                sound.play()

            except:  # isplaying = False
                pass
            if(thicc<16):
                thicc= thicc+2
            else:
                thicc=thicc-2
                if(thicc<2):
                    thicc=2
            cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
        cv2.imshow('Driver Drowsiness Detection v1.0',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
            
    cap.release()
    cv2.destroyAllWindows()

#display button
but1=Button(frame,padx=5,pady=5,width=39,bg='orange',fg='white',relief=GROOVE,command=webdetRec,text='Start Drowsiness Detection',font=('helvetica 15 bold'))
but1.place(x=5,y=104)
but2=Button(frame,padx=5,pady=5,width=39,bg='orange',fg='white',relief=GROOVE,command=root.destroy,text='Exit Program',font=('helvetica 15 bold'))
but2.place(x=5,y=184)

mainloop()
