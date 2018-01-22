from Tkinter import *
from PIL import Image
from PIL import ImageTk
import tkFileDialog
import cv2

global panelB
def select_image():

        global panelA, image1, imagef
 
	path = tkFileDialog.askopenfilename()

	if len(path) > 0:
		
		image = cv2.imread(path)
                image1 =image
                imagef = image
 
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
             
		image = Image.fromarray(image)
		
 
		image = ImageTk.PhotoImage(image)
		
		
		if panelA is None:
			
			panelA = Label(image=image)
			panelA.image = image
			panelA.pack(side="left", padx=10, pady=10)
 
			
		
		else:
			
			panelA.configure(image=image)
			panelA.image = image
			
def select_image1():
                global panelB
		blur = cv2.blur(image1,(5,5))
                blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB) 
 
		blur = Image.fromarray(blur)
 
		blur = ImageTk.PhotoImage(blur)
		
		if panelB is None:
			
			panelB = Label(image=blur)
			panelB.image = blur
			panelB.pack(side="right", padx=10, pady=10)
 
		
		else:
			
			panelB.configure(image=blur)
			panelB.image = blur

def select_image2():
	
	        global panelB
 
		blur = cv2.bilateralFilter(image1,9,75,75)
 
                blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB) 
		
		blur = Image.fromarray(blur)
 
		
		blur = ImageTk.PhotoImage(blur)
		
		if panelB is None:
			
			panelB = Label(image=blur)
			panelB.image = blur
			panelB.pack(side="right", padx=10, pady=10)
 
		
		else:
			
			panelB.configure(image=blur)
			panelB.image = blur
def select_image3():
	        global panelB
 
                kernel = np.ones((5,5),np.uint8)
                blackhat = cv2.morphologyEx(image1, cv2.MORPH_BLACKHAT, kernel)
 
                blackhat = cv2.cvtColor(blackhat, cv2.COLOR_BGR2RGB) 

		blackhat = Image.fromarray(blackhat)
 
		blackhat = ImageTk.PhotoImage(blackhat)
		
		if panelB is None:
			
			panelB = Label(image=blackhat)
			panelB.image = blackhat
			panelB.pack(side="right", padx=10, pady=10)
 
		else:
			
			panelB.configure(image=blackhat)
			panelB.image = blackhat
def select_image4():
	
	        global panelB
 
		gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
		edged = cv2.Canny(gray, 50, 100)
 
		
		edged = Image.fromarray(edged)
 
		
		edged = ImageTk.PhotoImage(edged)
		
		if panelB is None:
			
		
			panelB = Label(image=edged)
			panelB.image = edged
			panelB.pack(side="right", padx=10, pady=10)
 
		else:
			
			panelB.configure(image=edged)
			panelB.image = edged
def select_image5():

	        global panelB
 
                kernel = np.ones((5,5),np.uint8)
                closing = cv2.morphologyEx(image1, cv2.MORPH_CLOSE, kernel)
 
                closing = cv2.cvtColor(closing, cv2.COLOR_BGR2RGB) 

		closing = Image.fromarray(closing)
 
		
		closing = ImageTk.PhotoImage(closing)
		
		if panelB is None:
			
			panelB = Label(image=closing)
			panelB.image = closing
			panelB.pack(side="right", padx=10, pady=10)
 
		
		else:
			
			panelB.configure(image=closing)
			panelB.image = closing
def select_image6():
	
	        global panelB
 
	 
                kernel = np.ones((5,5),np.uint8)
                dilation = cv2.dilate(image1,kernel,iterations = 1)
 
                dilation = cv2.cvtColor(dilation, cv2.COLOR_BGR2RGB) 
		
		dilation = Image.fromarray(dilation)
 
		dilation = ImageTk.PhotoImage(dilation)
		
		if panelB is None:
			
 
			panelB = Label(image=dilation)
			panelB.image = dilation
			panelB.pack(side="right", padx=10, pady=10)
 
		
		else:
			
			panelB.configure(image=dilation)
			panelB.image = dilation
def select_image7():
	
	        global panelB
 
                kernel = np.ones((5,5),np.uint8)
                erosion = cv2.erode(image1,kernel,iterations = 1)
 
                erosion = cv2.cvtColor(erosion, cv2.COLOR_BGR2RGB) 
		
		erosion = Image.fromarray(erosion)
 
		erosion = ImageTk.PhotoImage(erosion)
		
		if panelB is None:
			
			
			panelB = Label(image=erosion)
			panelB.image = erosion
			panelB.pack(side="right", padx=10, pady=10)
 
		
		else:
			
			panelB.configure(image=erosion)
			panelB.image = erosion
def select_image8():
	        global panelB
                image2 = imagef
                face_cascade = cv2.CascadeClassifier('/home/georgiana/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
                eye_cascade = cv2.CascadeClassifier('/home/georgiana/opencv/data/haarcascades/haarcascade_eye.xml')
                gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x,y,w,h) in faces:
                     image2 = cv2.rectangle(image2,(x,y),(x+w,y+h),(255,0,0),2)
                     roi_gray = gray[y:y+h, x:x+w]
                     roi_color = image2[y:y+h, x:x+w]
                     eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex,ey,ew,eh) in eyes:
                     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

                image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB) 
		
		image2 = Image.fromarray(image2)
		
 
		
		image2 = ImageTk.PhotoImage(image2)
		
		if panelB is None:
			
			panelB = Label(image=image2)
			panelB.image = image2
			panelB.pack(side="left", padx=10, pady=10)

 
		
		else:
			
			panelB.configure(image=image2)
			panelB.image = image2
def select_image9():
	
	        global panelB
 
		blur = cv2.GaussianBlur(image1,(5,5),0)
 
                blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB) 
 
		blur = Image.fromarray(blur)
 
		blur = ImageTk.PhotoImage(blur)
		
		if panelB is None:
			
			panelB = Label(image=blur)
			panelB.image = blur
			panelB.pack(side="right", padx=10, pady=10)
 
		
		else:
			
			panelB.configure(image=blur)
			panelB.image = blur
def select_image10():
	
	        global panelB
 
	
                kernel = np.ones((5,5),np.uint8)
                gradient = cv2.morphologyEx(image1, cv2.MORPH_GRADIENT, kernel)
 
                gradient = cv2.cvtColor(gradient, cv2.COLOR_BGR2RGB) 

		gradient = Image.fromarray(gradient)
 
		gradient = ImageTk.PhotoImage(gradient)
		
		if panelB is None:
			
 
			panelB = Label(image=gradient)
			panelB.image = gradient
			panelB.pack(side="right", padx=10, pady=10)
 
		else:
			
			panelB.configure(image=gradient)
			panelB.image = gradient
def select_image11():
	
	        global panelB
 
		laplacian = cv2.Laplacian(image1,cv2.CV_64F)
 
		
		laplacian = Image.fromarray(laplacian.astype('uint8'))
 
		laplacian = ImageTk.PhotoImage(laplacian)
		
		if  panelB is None:
			
			panelB = Label(image=laplacian)
			panelB.image = laplacian
			panelB.pack(side="right", padx=10, pady=10)
 
		else:
			
			panelB.configure(image=laplacian)
			panelB.image = laplacian
def select_image12():
	
	        global panelB
 
		median = cv2.medianBlur(image1,5)
 
                median = cv2.cvtColor(median, cv2.COLOR_BGR2RGB) 
 
		median = Image.fromarray(median)
 
		median = ImageTk.PhotoImage(median)
		
		if panelB is None:
			
			panelB = Label(image=median)
			panelB.image = median
			panelB.pack(side="right", padx=10, pady=10)
 
		else:
			
			panelB.configure(image=median)
			panelB.image = median
def select_image13():
	
	        global panelB
 
                kernel = np.ones((5,5),np.uint8)
                opening = cv2.morphologyEx(image1, cv2.MORPH_OPEN, kernel)
 
                opening = cv2.cvtColor(opening, cv2.COLOR_BGR2RGB) 
		
		opening = Image.fromarray(opening)
 
		opening = ImageTk.PhotoImage(opening)
		
		if panelB is None:
			
			panelB = Label(image=opening)
			panelB.image = opening
			panelB.pack(side="right", padx=10, pady=10)
 
		else:
			
			panelB.configure(image=opening)
			panelB.image = opening
def select_image14():
	
	        global panelB
 
		sobelx = cv2.Sobel(image1,cv2.CV_64F,1,0,ksize=5)
 
		sobelx = Image.fromarray(sobelx.astype('uint8'))
 
		sobelx = ImageTk.PhotoImage(sobelx)
		
		if panelB is None:
			
			panelB = Label(image=sobelx)
			panelB.image = sobelx
			panelB.pack(side="right", padx=10, pady=10)
 
		else:
			
			panelB.configure(image=sobelx)
			panelB.image = sobelx
def select_image15():
	
	        global panelB
	
		sobely = cv2.Sobel(image1,cv2.CV_64F,0,1,ksize=5)
 
		sobely = Image.fromarray(sobely.astype('uint8'))
 
		sobely = ImageTk.PhotoImage(sobely)
		
		if panelB is None:
			
			panelB = Label(image=sobely)
			panelB.image = sobely
			panelB.pack(side="right", padx=10, pady=10)
 
		else:
			
			panelB.configure(image=sobely)
			panelB.image = sobely
def select_image16():
	
	        global panelB
 
                kernel = np.ones((5,5),np.uint8)
                tophat = cv2.morphologyEx(image1, cv2.MORPH_TOPHAT, kernel)
 
                tophat = cv2.cvtColor(tophat, cv2.COLOR_BGR2RGB) 
		
		tophat = Image.fromarray(tophat)
 
		tophat = ImageTk.PhotoImage(tophat)
		
		if panelB is None:
			
 
			panelB = Label(image=tophat)
			panelB.image = tophat
			panelB.pack(side="right", padx=10, pady=10)

		else:
			
		
			panelB.configure(image=tophat)
			panelB.image = tophat



root = Tk()
panelA = None
panelB = None
 
btn1 = Button(root, text="Select an image", command=select_image)
btn1.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
btn2 = Button(root, text="Averaging", command=select_image1)
btn2.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
btn3 = Button(root, text="Bilateral", command=select_image2)
btn3.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
btn4 = Button(root, text="Blackhat", command=select_image3)
btn4.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
btn5 = Button(root, text="Cany", command=select_image4)
btn5.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
btn6 = Button(root, text="Closing", command=select_image5)
btn6.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
btn7 = Button(root, text="Dilation", command=select_image6)
btn7.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
btn8 = Button(root, text="Erosion", command=select_image7)
btn8.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
btn9 = Button(root, text="Facedetection", command=select_image8)
btn9.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
btn10 = Button(root, text="Gaussian", command=select_image9)
btn10.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
btn11 = Button(root, text="Gradient", command=select_image10)
btn11.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
btn12 = Button(root, text="Laplacian", command=select_image11)
btn12.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
btn13 = Button(root, text="Median", command=select_image12)
btn13.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
btn14 = Button(root, text="Opening", command=select_image13)
btn14.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
btn15 = Button(root, text="SobelX", command=select_image14)
btn15.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
btn16 = Button(root, text="SobelY", command=select_image15)
btn16.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
btn17 = Button(root, text="Tophat", command=select_image16)
btn17.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
 
root.mainloop()

