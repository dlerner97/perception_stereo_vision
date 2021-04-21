import cv2
import json
import numpy as np
from time import time
from os import system
from numpy.lib.function_base import disp
from scipy.signal import convolve2d
from video_handler import VideoHandler
import matplotlib.pyplot as plt

class StereoVision:
    def __init__(self, cam_params, img0, img1) -> None:
        self.img0 = img0
        self.img1 = img1
        self.cam_params = cam_params
        self.img_count = 0

    def get_matches(self, img0=None, img1=None, num_matches=50, debug=False):
        if img0 is None:
            img0 = self.img0
        if img1 is None:
            img1 = self.img1
        
        sift = cv2.SIFT_create()
        
        keypts_0, hog_0 = sift.detectAndCompute(img0, None)
        keypts_1, hog_1 = sift.detectAndCompute(img1, None)  
        
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = bf.match(hog_0, hog_1)
        matches = sorted(matches, key = lambda x: x.distance)
        
        if debug:
            match_img = cv2.drawMatches(img0, keypts_0, img1, keypts_1, matches[:num_matches], img1, flags=2)
            cv2.imshow(f"img{self.img_count}", cv2.drawKeypoints(img0, keypts_0, img0))
            self.img_count += 1
            cv2.imshow(f"img{self.img_count}", cv2.drawKeypoints(img1, keypts_1, img1))
            self.img_count += 1
            cv2.imshow("matches", match_img)
            cv2.waitKey(0)
         
        if num_matches <= 0:
            return matches, keypts_0, keypts_1  
        return matches[:num_matches], keypts_0, keypts_1
    
    def get_cameras_rel(self, match_info):
        matches, keypts_0, keypts_1 = match_info
        
        A = []
        self.sorted_keypts_0 = []
        self.sorted_keypts_1 = []
        for match in matches:
            u, v = keypts_0[match.queryIdx].pt
            u_p, v_p = keypts_1[match.trainIdx].pt
            A.append([u_p*u, u_p*v, u_p, v_p*u, v_p*v, v_p, u, v, 1])
            
            self.sorted_keypts_0.append((u,v))
            self.sorted_keypts_1.append((u_p, v_p))
                
        self.sorted_keypts_0 = np.int32(self.sorted_keypts_0)
        self.sorted_keypts_1 = np.int32(self.sorted_keypts_1)
            
        A = np.array(A)
        
        # F, self.mask = cv2.findFundamentalMat(self.sorted_keypts_0, self.sorted_keypts_1, cv2.RANSAC)
        
        _, _, V = np.linalg.svd(A)
        F = V[-1, :]
        F = (F/F[-1]).reshape(3,3)
        
        self.K0 = np.array(self.cam_params["cam0"])
        self.K1 = np.array(self.cam_params["cam1"])  
        E = self.K1.T @ F @ self.K0
        E = E/E[-1, -1]
        
        U, S, V = np.linalg.svd(E)
        W = np.array([[0,-1,0], [1,0,0], [0,0,1]])
        t = U @ W @ S @ U.T
        R = U @ W.T @ V.T
        return R, t, F
      
    def rectify_images(self):
        matches, keypts_0, keypts_1 = self.get_matches()
        R, t, F = self.get_cameras_rel((matches, keypts_0, keypts_1))
        _, H0, H1 = cv2.stereoRectifyUncalibrated(self.sorted_keypts_0, self.sorted_keypts_1, F, self.img0.shape[:2][::-1])
        print(10*"-", "H0", 10*"-")
        print(H0)
        print(24*"-")
        
        print(10*"-", "H1", 10*"-")
        print(H1)
        print(24*"-")
        
        # self.disp_epipoles(self.img0, self.img1, F)
        F = (np.linalg.inv(H1)).T @ F @ np.linalg.inv(H0)
        keypts_0 = np.array([self.sorted_keypts_0], dtype=np.float32)
        keypts_1 = np.array([self.sorted_keypts_1], dtype=np.float32)
        self.sorted_keypts_0 = cv2.perspectiveTransform(keypts_0, H0)[0].astype(np.int32, copy=False)
        self.sorted_keypts_1 = cv2.perspectiveTransform(keypts_1, H1)[0].astype(np.int32, copy=False)
        
        img0 = cv2.warpPerspective(self.img0, H0, self.img0.shape[:2][::-1])
        img1 = cv2.warpPerspective(self.img1, H1, self.img1.shape[:2][::-1])
        # self.disp_epipoles(img0, img1, F)
        return img0, img1, F
        
    def apply_window_matching(self, window_size=19, disparity_thresh=500, debug=False):
        img0, img1, F = self.rectify_images()
        epilines0 = cv2.computeCorrespondEpilines(self.sorted_keypts_0.reshape(-1,1,2), 2, F).reshape(-1,3)
        y_vals_0 = [int(round(-line_wt[2]/line_wt[1])) for line_wt in epilines0]
        y_vals_0.sort()
        
        epilines1 = cv2.computeCorrespondEpilines(self.sorted_keypts_1.reshape(-1,1,2), 2, F).reshape(-1,3)
        y_vals_1 = [int(round(-line_wt[2]/line_wt[1])) for line_wt in epilines1]
        y_vals_1.sort()
        
        win_r = window_size//2
               
        img0_gray = cv2.GaussianBlur(cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY), (15, 15), 0)
        img1_gray = cv2.GaussianBlur(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), (15, 15), 0)
        s0 = img0_gray.shape[1]
        s1 = img1_gray.shape[1]
  
        plt.figure()
        disparity = []
        wait = 1
        sy = min([img0_gray.shape[0], img0_gray.shape[1]])
        print("0%")
        for y in range(win_r+1, sy-(win_r+1+1)):
            disparity.append([])
            for px0 in range(win_r, s0-(win_r+1)):

                kernel = img0_gray[y-win_r:y+win_r+1, px0-win_r:px0+win_r+1].astype(int)[::-1, ::-1]
                kernel = (kernel - np.mean(kernel))/(np.std(kernel)+.01)

                img_segment = img1_gray[y-win_r-1:y+win_r+1+1, :]
                seg_mean = np.mean(img_segment)
                seg_std = np.std(img_segment)
                
                img_segment = (img_segment - seg_mean)/(seg_std+.01)
                res = convolve2d(img_segment, kernel, mode='valid')
                px1 = (np.argmax(res) + win_r + 1)%s0
                
                disparity[-1].append(abs(px1-px0))
                
                if debug:
                    res_norm = np.round(255*(res-min(res))/(max(res)-min(res))).astype(np.uint8)
                    res_norm_ls = np.vstack((res_norm, res_norm, res_norm, res_norm, res_norm, res_norm, res_norm))
                    
                    img0_gray_cp = img0_gray.copy()
                    img1_gray_cp = img1_gray.copy()
                    cv2.circle(img0_gray_cp, (px0, y), 1, 150, -1)
                    cv2.rectangle(img0_gray_cp, (px0-win_r, y-win_r), (px0+win_r, y+win_r), 150, 1)
                    img1_gray_cp[y-len(res_norm_ls)//2:y+len(res_norm_ls)//2+1, win_r:s1-win_r] = res_norm
                                
                    cv2.imshow("gray0", img0_gray_cp)
                    cv2.imshow("color map", img1_gray_cp)
                    
                    k = cv2.waitKey(wait)
                    if k == ord('q'):
                        exit()
                    elif k == ord('w'):
                        wait = 1 - wait
                    elif k == ord('e'):
                        print(kernel)
                        print(img_segment)
                        plt.plot(res)
                        plt.show()

            print ("\033[A                             \033[A")
            per_done = 100*y/(sy-(win_r+1))
            print(round(per_done/2)*"|", "  ", f"{round(per_done, 2)}%")
           
        print ("\033[A                             \033[A")
        print(50*"|", "  ", "100% Finished!")
        
        if debug:
            from pandas import DataFrame
            print(DataFrame(disparity))
        
        disparity = np.array(disparity)
        min_corr = np.min(disparity)
        max_corr = np.max(disparity)
        disparity_map = np.round(255-255*(disparity.ravel() - min_corr)/(max_corr - min_corr)).astype(np.uint8).reshape(disparity.shape)
        cv2.imshow("disparity map", disparity_map)
        
        disparity[disparity > disparity_thresh] = disparity_thresh
        
        if debug:
            print(DataFrame(disparity))
            
        min_corr = np.min(disparity)
        max_corr = np.max(disparity)
        disparity_map_thresh = np.round(255-255*(disparity.ravel() - min_corr)/(max_corr - min_corr)).astype(np.uint8).reshape(disparity.shape)
        cv2.imshow("thresholded disparity map", disparity_map_thresh)
        
        self.disp_heat_map(disparity_map)
        self.disp_heat_map(disparity_map_thresh)
        
        baseline = self.cam_params["baseline"]
        focal_len = self.cam_params["cam0"][0][0]
        print(focal_len)
        
        
        cv2.waitKey(0)
        
                 
    def disp_heat_map(self, img):        
        colormap = plt.get_cmap('jet')
        heatmap = (colormap(img) * 2**16).astype(np.uint16)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        cv2.imshow(f"heat map {self.img_count}", heatmap)
        self.img_count += 1

    def disp_epipoles(self, img0, img1, F):   
        img0 = img0.copy()
        img1 = img1.copy()
            
        def drawlines(img,lines,pts):
            r,c,_ = img.shape
            for r,pt in zip(lines,pts):
                color = tuple(np.random.randint(0,255,3).tolist())
                x0,y0 = map(int, [0, -r[2]/r[1] ])
                x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
                cv2.line(img, (x0,y0), (x1,y1), color,1)
                cv2.circle(img,tuple(pt),5,color,-1)
            return img
        
        epilines1 = cv2.computeCorrespondEpilines(self.sorted_keypts_1.reshape(-1,1,2), 2, F)
        epilines1 = epilines1.reshape(-1,3)
        img0_ = drawlines(img0,epilines1,self.sorted_keypts_0)

        epilines2 = cv2.computeCorrespondEpilines(self.sorted_keypts_0.reshape(-1,1,2), 1, F)
        epilines2 = epilines2.reshape(-1,3)
        img1_ = drawlines(img1, epilines2,self.sorted_keypts_1)
        
        cv2.imshow(f"img{self.img_count}", img0_)
        self.img_count += 1
        cv2.imshow(f"img{self.img_count}", img1_)
        self.img_count += 1
        cv2.waitKey(0)
        
if __name__ == "__main__":
    system('cls')
    
    start_time = time()
    cam_params = {}
    with open("cam_params.json", 'r') as file:
        cam_params = json.load(file)
        
    ds = 1
    img0 = cv2.imread(f"Dataset {ds}/im0.png")
    img1 = cv2.imread(f"Dataset {ds}/im1.png")
    img0 = VideoHandler.resize_frame(img0, 25)
    img1 = VideoHandler.resize_frame(img1, 25)
   
    stereo_vis = StereoVision(cam_params[f"d{ds}"], img0, img1)
    stereo_vis.apply_window_matching()
    
    # Calculate time elapsed
    time_elapsed_s = time() - start_time
    time_elapsed_mins = time_elapsed_s//60
    time_elapsed_hrs = time_elapsed_s//60**2
    time_elapsed_secs = time_elapsed_s%60
    
    print(f"The stereo vision execution took {time_elapsed_hrs} hrs, {time_elapsed_mins} mins, and {time_elapsed_secs} s to implement.\n")
