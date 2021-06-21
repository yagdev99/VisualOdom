import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import os

plt.ion()

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params,search_params)

files = os.listdir('data')
files = [file for file in files if len(file.split('.')) == 2 and file.split('.')[1] == 'png']
files.sort()
files.reverse()

K_mat = np.array([9.037596e+02, 0.000000e+00, 6.957519e+02, 0.000000e+00, 9.019653e+02, 2.242509e+02, 0.000000e+00, 0.000000e+00, 1.000000e+00])
K_mat = K_mat.reshape(3,3)
fx = K_mat[0][0]
cx = K_mat[0][-1]
cy = K_mat[1][-1]

num_called = 0

trajectory = []

sift = cv.SIFT_create()

if len(files) == 0:
    print('Error Opening File!')

else:
    file = files.pop()
    print(file)
    prev_rgb = rgb = cv.imread('data/' + file)
    kp, des = sift.detectAndCompute(rgb,None)
    prev_kp, prev_des = kp, des
    while len(files) != 0:
        
        prev_rgb = rgb
        file = files.pop()
        print(file)
        rgb = cv.imread('data/' + file)

        prev_kp = kp
        kp, des = sift.detectAndCompute(rgb,None)

        if prev_des is not None and len(prev_des)>2 and des is not None and len(des)>2:

            matches = flann.knnMatch(prev_des,des,k=2)

            matchesMask = [[0,0] for i in range(len(matches))]
            pt1 = list()
            pt2 = list()

            flag = 0

            try:
                for m,n in matches:
                    if m.distance < 0.75*n.distance:
                        x1, y1 = prev_kp[m.queryIdx].pt
                        x2, y2 = kp[m.trainIdx].pt

                        pt1.append([x1,y1])
                        pt2.append([x2,y2])
            except:
                flag = 1

            if flag == 0:
                num_called += 1
                pt1 = np.asarray(pt1, dtype = np.float32)
                pt2 = np.asarray(pt2, dtype = np.float32)
                E, mask = cv.findEssentialMat(pt1, pt2,cameraMatrix = K_mat, method=cv.RANSAC, prob=0.80, threshold=1)
                pt1 = pt1[mask.ravel() == 1]
                pt2 = pt2[mask.ravel() == 1]
                _, R, t, mask = cv.recoverPose(E, pt1, pt2,K_mat)


                R = R.T
                t = -1*np.matmul(R, t)

                if num_called == 1:
                    prev_R = curr_R = R
                    prev_t = pose = t
                else:
                    prev_R = curr_R
                    curr_R = np.matmul(prev_R, R)
                    
                    prev_t = pose
                    pose = np.matmul(prev_R, t) + prev_t

                prev_R = curr_R
                prev_t = pose

                trajectory.append(pose)
                traj = np.asarray(trajectory).reshape(-1,3)
                traj = traj.T
                np.save('np',traj)

                plt.subplot(2,1,1)
                plt.plot(traj[0][:],traj[2][:])
                plt.subplot(2,1,2)
                plt.imshow(rgb)

                plt.pause(0.01)
                plt.show()
            
        

            




