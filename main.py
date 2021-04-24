import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import glob
import numba as nb
from multiprocessing import Pool
import time
from segment import Segment


@nb.jit
def vertical_match(upper,lower):
    matches=[]
    for u in upper:
        for l in lower:
            if u.ontop(l):
                matches.append((u,l))
    return matches

@nb.jit
def label_iterative(x,y,img,display_img,R,G,B):
    frontier = []
    frontier.append((x,y))
    segment = Segment(img.shape[0],0,img.shape[1],0,B)
    
    while len(frontier) != 0:
        x,y = frontier.pop(-1)
        if x<0 or y<0 or x >= img.shape[0] or y >= img.shape[1]: continue
        if not img[x,y]==255: continue
        
        segment.update(x,y)
        img[x,y]=0
        display_img[x,y,0]=R
        display_img[x,y,1]=G
        display_img[x,y,2]=B
        
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                if i==0 and j == 0: continue
                frontier.append((x+i,y+j))
                
    return segment

def labelfunc(img,tresh_bottom,tresh_up):
    new_img = np.zeros((img.shape[0],img.shape[1],3), dtype=np.uint8)
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  
    img_hsv = cv2.inRange(img_hsv,tresh_bottom,tresh_up)      
    
    return label_loop(new_img,img_hsv)
   
@nb.jit  
def label_loop(new_img,img_hsv):
    B=0
    segments = []
    for x in range(img_hsv.shape[0]):
        for y in range(img_hsv.shape[1]):
            if img_hsv[x,y]==255:
                G = random.randint(0,255)
                R = random.randint(0,255)
                B = B+1
                segments.append(label_iterative(x,y,img_hsv,new_img,R,G,B))
     
    segments = list(filter(lambda x: x.size > 200,segments))        
    return new_img, segments
            
def draw_match(img, match, scale):
    cv2.rectangle(img,(int(match[0].y_min*scale),int(match[0].x_min*scale)),(int(match[0].y_max*scale),int(match[0].x_max*scale)),(0,255,0),3)
    
    if match[1].y_min+10 < match[0].y_min:
        up_left = (int((match[0].y_min-10)*scale),int(match[1].x_min*scale))
    else:
        up_left = (int(match[1].y_min*scale),int(match[1].x_min*scale))
        
    if match[1].y_max > 10 + match[0].y_max:    
        down_right = (int((match[0].y_max+10)*scale),int(match[1].x_max*scale))
    else:
        down_right = (int(match[1].y_max*scale),int(match[1].x_max*scale))
        
    cv2.rectangle(img, up_left,down_right,(0,0,255),3)
                
def find(imdir, thresholds, training = False):
    start = time.time()
    
    img = cv2.imread(imdir)
    orig_res=img.shape
    img = cv2.medianBlur(img,15)
    
    img = cv2.resize(img,(800,450))
    img_cpy = np.copy(img)
    img_matches = cv2.cvtColor(cv2.imread(imdir),cv2.COLOR_BGR2RGB)
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  
    
    img_pants, segment_pants = labelfunc(img,thresholds[0],thresholds[1])
    img_torso, segment_torso = labelfunc(img_cpy,thresholds[2],thresholds[3])
    matches = vertical_match(segment_torso,segment_pants)
    
    print(time.time()-start)
    
    if not training:
        """
        print("torso")
        list(map(lambda x: print(str(x)),segment_torso))
        print("pants")
        list(map(print,segment_pants))
        print("matches")
        list(map(print,matches))
        """
        
        for m in matches:
            draw_match(img_matches,m,orig_res[1]/800)
        
        plt.imshow(img_matches)
        plt.show()
        
        fig, axs = plt.subplots(2,2)
        axs[0,0].imshow(img_pants)
        axs[0,1].imshow(img_torso)
        axs[1,0].imshow(img_matches)
        axs[1,1].imshow(img_hsv)
        plt.show()

if __name__ == "__main__":
    labels = glob.glob("sample_images\\*")
    
    for i in range(4):
        find(labels[i],thresholds=[(10,15,30),(30,70,137),(10,140,60),(80,255,255)])