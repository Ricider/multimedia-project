import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import glob
import numba as nb
import time
from segment import Segment
from statistics import mean
from itertools import product
from multiprocessing import Pool

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
        
        for i in range(-3,4):
            for j in range(-3,4):
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
       
training_cache_p = {}
training_cache_t = {}
def find(img, thresholds, video = False, training = False):
    if not training: start = time.time()
    
    if not (training and str((thresholds[0],thresholds[1])) in training_cache_p and str((thresholds[2],thresholds[3])) in training_cache_t):
        if not video: img_matches = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        orig_res=img.shape
        
        img = cv2.resize(img,(800,450))
        img_cpy = np.copy(img)
        
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  
    
    if training and str((thresholds[0],thresholds[1])) in training_cache_p:
        segment_pants = training_cache_p[str((thresholds[0],thresholds[1]))]
    else:
        img_pants, segment_pants = labelfunc(img,thresholds[0],thresholds[1])
        if training: training_cache_p[str((thresholds[0],thresholds[1]))] = segment_pants
    
    if training and str((thresholds[2],thresholds[3])) in training_cache_t:
        segment_torso = training_cache_t[str((thresholds[2],thresholds[3]))]
    else:
        img_torso, segment_torso = labelfunc(img_cpy,thresholds[2],thresholds[3])
        if training: training_cache_t[str((thresholds[2],thresholds[3]))] = segment_torso

    matches = vertical_match(segment_torso,segment_pants)
    
    if not training: print(time.time()-start)
    
    if training:
        return matches,segment_torso,segment_pants
    elif video:
        return matches
    else:
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
        return matches

def vid_read(viddir,thresholds):
    find_matches = lambda x: find(x,thresholds=thresholds,video=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4',fourcc, 48.0, (1920,1080))    
    cap = cv2.VideoCapture(viddir)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        matches = find_matches(frame)
        for m in matches:
            draw_match(frame,m,frame.shape[1]/800) 
        
        out.write(frame)
        cv2.imshow('frame',frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break     
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()        

percentile_cache_t = {}
percentile_cache_p = {}
def cached_percentile(pixels,cutoff,torso):
    if torso and cutoff in percentile_cache_t:
        return percentile_cache_t[cutoff]
    elif (not torso) and cutoff in percentile_cache_p:
        return percentile_cache_p[cutoff]
    
    percentiles = np.percentile(pixels,cutoff,axis=0)
    if torso:
        percentile_cache_t[cutoff] = percentiles
    else:
        percentile_cache_p[cutoff] = percentiles
    return percentiles

def avg_thresh(img, ground_truth, percentile, train_size = 10, precision = 2):
    torso_pixels=[]
    pants_pixels=[]
    
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if ground_truth[y,x,2] > 100:
                torso_pixels.append(hsv_img[y,x])
            elif ground_truth[y,x,0] > 100:
                pants_pixels.append(hsv_img[y,x]) 
    
    best_matches = find(ground_truth,((110,10,10),(130,255,255),(0,10,10),(5,255,255)))    
    best_score = -1
    best_thresh = None
    bad_p_quartiles = []
    bad_t_quartiles = []
    old_v=0
    
    for i,v,k,j in product(list(range(percentile,percentile+train_size,precision)), repeat=4):
        if (i,v) in bad_p_quartiles or (k,j) in bad_t_quartiles:
            continue
        
        current_thresh = [cached_percentile(pants_pixels,i,torso=False),
                          cached_percentile(pants_pixels,100-v,torso=False),
                          cached_percentile(torso_pixels,k,torso=True),
                          cached_percentile(torso_pixels,100-j,torso=True)]

        current_matches,segment_torso,segment_pants = find(img.copy(),current_thresh, training=True)
               
        if len(segment_pants) <= best_score:
            bad_p_quartiles.append((i,v))
            continue
        if len(segment_torso) <= best_score:
            bad_t_quartiles.append((k,j))
            continue
            
        current_score = 0
        match_positions = [] #to penalize double matches
        for match in current_matches:
            match_found=False
            match_x = (match[0].x_max + match[0].x_min) / 2
            match_y = (match[0].y_max + match[0].y_min) / 2
            if (match_x,match_y) in match_positions:
                current_score -= 0.5
                continue
            match_positions.append((match_x,match_y))
            
            for true_match in best_matches:
                if match_x > true_match[0].x_min and match_x < true_match[0].x_max and match_y > true_match[0].y_min and match_y < true_match[0].y_max:
                    current_score+=1
                    match_found = True
                    break
                
            if not match_found:
                current_score -= 2 #penalize wrong matches
                
        if old_v != v:
            print("best thresh =",best_thresh,"\ni =", i,"v =", v,"\nbest score =",best_score)
            old_v = v
                
        if current_score > best_score:
            best_thresh = current_thresh
            best_score = current_score
        
    print(best_thresh)    
    print(best_score)
    return best_thresh

if __name__ == "__main__":
    
    vid_id = 1
    
    img = cv2.imread("sample_images\\{}.jpg".format(vid_id))
    truth = cv2.imread("sample_truths\\{}.jpg".format(vid_id))
    
    thresholds = avg_thresh(img.copy(),truth,5,25,1)
    #thresholds = [(0,7,17),(22,70,68),(13,117,44),(19,216,183)]
    find(img.copy(),thresholds)
    
    #[array([ 8., 21., 24.]), array([30., 81., 74.]), array([ 13., 119.,  45.]), array([ 18., 197., 158.])]
    #[array([ 6., 18., 22.]), array([110.,  89.,  80.]), array([ 13., 119.,  45.]), array([ 18., 188., 144.])]
    #thresholds = [(10,15,30),(30,70,137),(10,140,60),(80,255,255)]
    #advanced thresh = [(0,7,17),(22,70,68),(13,117,44),(19,216,183)]
    #probably bad bu worth a shot [array([ 6., 20., 18.]), array([165.,  84.,  67.]), array([ 12., 119.,  40.]), array([ 19., 213., 160.])]
    
    vid_read("sample_videos\\{}.mkv".format(vid_id),thresholds)
    