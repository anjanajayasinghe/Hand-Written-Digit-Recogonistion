import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2 

WINDOWSIZEX =640
WINDOWSIZEY = 480

BOUNDRYINC=5
WHITE=(255,255,255)
BLACK=(0,0,0)
RED=(255,0,0)

IMAGESAVE=False
img_cnt=1

MODEL=load_model("bestmodel.h5")

LABELS = {0:"Zero",1:"One",2:"Two",3:"Three",4:"Four",5:"Five",6:"Six",7:"Seven",8:"Eight",9:"Nine"}

#initailize our pygame
pygame.init()

FONT=pygame.font.Font(size=20)
DISPLAYSURF=pygame.display.set_mode((WINDOWSIZEX,WINDOWSIZEY))
WHITE_INT = DISPLAYSURF.map_rgb(WHITE)
pygame.display.set_caption("Numerical Identifier")

iswriting=False
PREDICT=True

number_xcord=[]
number_ycord=[]

while True:
   
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        
        if event.type == MOUSEMOTION and iswriting:
            xcord,ycord=event.pos
            pygame.draw.circle(DISPLAYSURF,WHITE,(xcord,ycord),4,0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting=True

        if event.type==MOUSEBUTTONUP:
            iswriting=False
            
            number_xcord=sorted(number_xcord)
            number_ycord=sorted(number_ycord)

            rect_min_x, rect_max_x= max(number_xcord[0]-BOUNDRYINC,0),min(WINDOWSIZEX,number_xcord[-1]+BOUNDRYINC)
            rect_min_y, rect_max_y= max(number_ycord[0]-BOUNDRYINC,0 ),min(number_ycord[-1]+BOUNDRYINC,WINDOWSIZEX)
    
            number_xcord=[]
            number_ycord=[]
            

            img_arr=np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x,rect_min_y:rect_max_y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite("image.png")
                img_cnt+=1
            
            if PREDICT:
                image=cv2.resize(img_arr,(28,28))
                image=np.pad(image,(10,10),"constant",constant_values=0)
                image=cv2.resize(image,(28,28))/255

                label=str(LABELS[np.argmax(MODEL.predict(image.reshape(1,28,28,1)))])
                border_thickness = 2  
                pygame.draw.rect(DISPLAYSURF, RED, (rect_min_x, rect_min_y, (rect_max_x-rect_min_x), (rect_max_y-rect_min_y)),border_thickness)

                textsurface=FONT.render(label,True,RED,WHITE)
                textRexObj=textsurface.get_rect()
                textRexObj.left,textRexObj.bottom=rect_min_x,rect_max_y

                DISPLAYSURF.blit(textsurface,textRexObj)
            
            elif event.type==pygame.KEYDOWN:
                if event.key==pygame.K_a:
                    DISPLAYSURF.fill(WHITE)

    pygame.display.update()
