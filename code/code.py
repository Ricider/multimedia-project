import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import queue

class Segment:
    def __init__(self,x_min,x_max,y_min,y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.size=0
    def update(self,x,y):
        self.x_min = min(self.x_min,x)
        self.x_max = max(self.x_max,x)
        self.y_min = min(self.y_min,y)
        self.y_max = max(self.y_max,y)
        self.size += 1
    def ontop(self,other):
        horizontally_alligned = self.y_min + 15 > other.y_min and self.y_max < 15 + other.y_max
        length = (other.x_max-other.x_min)
        vertically_alligned = self.x_max + 15 > other.x_min and self.x_max < 15 + other.x_min
        return horizontally_alligned and vertically_alligned
    def __str__(self):
        return "x=({}-{}) y=({}-{}) size = {}".format(self.x_min,self.x_max,self.y_min,self.y_max,self.size)
    __repr__=__str__

def vertical_match(upper,lower):
    matches=[]
    for u in upper:
        for l in lower:
            if u.ontop(l):
                matches.append((u,l))
    return matches

def label_iterative(x,y,img,display_img,R,G,B):
    frontier = queue.Queue()
    frontier.put((x,y))
    segment = Segment(img.shape[0],0,img.shape[1],0)
    
    while not frontier.empty():
        x,y = frontier.get()
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
                frontier.put((x+i,y+j))
                
    return segment

def labelfunc(img,tresh_bottom,tresh_up):
    new_img = np.zeros((img.shape[0],img.shape[1],3), dtype=np.uint8)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  
    img_hsv = cv2.inRange(img_hsv,tresh_bottom,tresh_up)      
    segments = []
    
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img_hsv[x,y]==255:
                G = random.randint(0,255)
                R = random.randint(0,255)
                B = random.randint(0,255)
                segments.append(label_iterative(x,y,img_hsv,new_img,R,G,B))
     
    segments = list(filter(lambda x: x.size > 100,segments))        
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
                
if __name__ == "__main__":
    #imdir = "sample_images\\mirage_a_site.jpg"
    imdir = "sample_images\\long_door.jpg"
    
    img = cv2.imread(imdir)
    orig_res=img.shape
    img = cv2.medianBlur(img,15)
    
    img = cv2.resize(img,(800,450))
    img_cpy = np.copy(img)
    img_matches = cv2.cvtColor(cv2.imread(imdir),cv2.COLOR_BGR2RGB)
    
    img_pants, segment_pants = labelfunc(img,(10,15,30),(30,70,100))
    img_torso, segment_torso = labelfunc(img_cpy,(10,140,60),(20,255,255))
    matches = vertical_match(segment_torso,segment_pants)
    
    print("torso")
    list(map(print,segment_torso))
    print("pants")
    list(map(print,segment_pants))
    print("matches")
    list(map(print,matches))
   
    for m in matches:
        draw_match(img_matches,m,orig_res[1]/800)
    
    plt.imshow(img_matches)
    plt.show()
    
    fig, axs = plt.subplots(1,3)
    axs[0].imshow(img_pants)
    axs[1].imshow(img_torso)
    axs[2].imshow(img_matches)
    plt.show()