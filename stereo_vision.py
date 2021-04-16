import cv2
import json
import numpy as np
from os import system

from numpy.lib.function_base import copy
from video_handler import VideoHandler

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
            cv2.imshow("img{self.img_count}", cv2.drawKeypoints(img1, keypts_1, img1))
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
        
        self.disp_epipoles(self.img0, self.img1, F)
        F = (np.linalg.inv(H1)).T @ F @ np.linalg.inv(H0)
        keypts_0 = np.array([self.sorted_keypts_0], dtype=np.float32)
        keypts_1 = np.array([self.sorted_keypts_1], dtype=np.float32)
        self.sorted_keypts_0 = cv2.perspectiveTransform(keypts_0, H0)[0].astype(np.int32, copy=False)
        self.sorted_keypts_1 = cv2.perspectiveTransform(keypts_1, H1)[0].astype(np.int32, copy=False)
        
        img0 = cv2.warpPerspective(self.img0, H0, self.img0.shape[:2][::-1])
        img1 = cv2.warpPerspective(self.img1, H1, self.img1.shape[:2][::-1])
        self.disp_epipoles(img0, img1, F)
        return img0, img1, F
        
    def apply_window_matching(self):
        img0, img1, F = self.rectify_images()
        epilines0 = cv2.computeCorrespondEpilines(self.sorted_keypts_0.reshape(-1,1,2), 2, F).reshape(-1,3)
        y_vals_0 = [int(round(-line_wt[2]/line_wt[1])) for line_wt in epilines0]
        y_vals_0.sort()
        
        epilines1 = cv2.computeCorrespondEpilines(self.sorted_keypts_1.reshape(-1,1,2), 2, F).reshape(-1,3)
        y_vals_1 = [int(round(-line_wt[2]/line_wt[1])) for line_wt in epilines1]
        y_vals_1.sort()
        
        win_size = 9
        win_r = win_size//2
        s0 = img0.shape[1]
        s1 = img1.shape[1]
        correspondance = []
        for y0, y1 in zip(y_vals_0, y_vals_1):
            y0_min_x = []
            for px0 in range(s0):
                min_ssd = 1000
                x_ssd = 0
                window_0 = img0[px0-win_r:px0+win_r, y0]
                for px1 in range(s1):
                    window_1 = img1[px1-win_r:px1+win_r, y1]
                    ssd = sum([(v0-v1)**2 for v0, v1 in zip(window_0.ravel(), window_1.ravel())])
                    if ssd < min_ssd:
                        min_ssd = ssd
                        x_ssd = px1
                y0_min_x.append(x_ssd)
            correspondance.append(y0_min_x)
            print(y0)
        print(correspondance)            
        
    def disp_epipoles(self, img0, img1, F):       
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
    cam_params = {}
    with open("cam_params.json", 'r') as file:
        cam_params = json.load(file)
        
    ds = 1
    img0 = cv2.imread(f"Dataset {ds}/im0.png")
    img1 = cv2.imread(f"Dataset {ds}/im1.png")
    img0 = VideoHandler.resize_frame(img0, 25)
    img1 = VideoHandler.resize_frame(img1, 25)

    # cv2.imshow("img0", img0)
    # cv2.imshow("img1", img1)
    # cv2.waitKey(0)
    # print(img1.shape, img0.shape)
    
    stereo_vis = StereoVision(cam_params[f"d{ds}"], img0, img1)
    stereo_vis.apply_window_matching()
