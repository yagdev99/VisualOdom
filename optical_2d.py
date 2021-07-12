import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import os

plt.ion()

# Camera matrix
K_mat = np.array([9.037596e+02, 0.000000e+00, 6.957519e+02, 0.000000e+00, 9.019653e+02, 2.242509e+02, 0.000000e+00, 0.000000e+00, 1.000000e+00])
K_mat = K_mat.reshape(3,3)
fx = K_mat[0][0]
cx = K_mat[0][-1]
cy = K_mat[1][-1]


sift = cv.SIFT_create()

#Using CLAHE to Equalize histogram in order to improve contrast of image
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


num_called = 0
trajectory = []



def load_img(file, scale_xy = 0.8):
    """
    @brief Function to load, resize and apply CLAHE to the image

    @param file: File name of the image which is to be loaded

    @param scale_xy: Scale to resize the image by

    """

    img = clahe.apply(cv.cvtColor(cv.imread(path + '/' + file),cv.COLOR_BGR2GRAY))

    return cv.resize(img, (0,0), fx = scale_xy, fy = scale_xy)

# 
def find_kp(gray):
    """
    @brief function to find keypoints using SIFT feature detector and return an array of the respective coordinates in the image

    @param gray: image to find keypoints
    """

    new_kp = []

    kp = sift.detect(gray,None)

    for i in range(len(kp)):
        new_kp.append(kp[i].pt)

    return np.array(new_kp, dtype = np.float32).reshape(-1,1,2)

# 
def track_features(prev_gray, gray, kp):

    """
    @brief function to track features from prev_gray to gray using Lucas Kanade Optical Flow

    @param prev_gray: gray image of previous time instant

    @param gray: gray image of current time instant

    @param kp: keypoints of prev_gray to be tracked in gray
    """

    pt1, st, err = cv.calcOpticalFlowPyrLK(prev_gray.astype(np.uint8), gray.astype(np.uint8), kp, None, **lk_params)
    pt2, st, err = cv.calcOpticalFlowPyrLK(gray.astype(np.uint8), prev_gray.astype(np.uint8), pt1.astype(np.float32), None, **lk_params)

    good_new = pt1[st == 1].reshape(-1,2)
    good_old = pt2[st == 1].reshape(-1,2)

    if pt1 is not None and pt2 is not None:
        run = True
    else:
        run = False

    return good_old, good_new, run



def start_tracking(path, scale_xy):
    global trajectory, num_called, K_mat

    # Changing Intrinsic calibration matrix to account to 
    # resized image
    scale = scale_xy*np.eye(3)
    scale[-1][-1] = 1
    K_mat = np.matmul(scale,K_mat)


    # Creating a list of all files in dataset directory
    files = os.listdir(path)
    files = [file for file in files if len(file.split('.')) == 2 and file.split('.')[1] == 'png']
    files.sort()
    files.reverse()


    if len(files) == 0:
        print('Error Opening File!')

    else:
        file = files.pop()
        # print(file)
        prev_gray = gray = load_img(file,scale_xy)
        
        kp = find_kp(gray)

        # Loop to iterate over all images
        while len(files) != 0:

            prev_gray = gray

            file = files.pop()

            gray = load_img(file, scale_xy)

            # Detects 
            if len(kp) < 2000:
                kp = find_kp(prev_gray)

            good_old, good_new, run= track_features(prev_gray, gray, kp)

            if run == True:

                num_called += 1

                F,mask = cv.findFundamentalMat(good_old, good_new, cv.FM_LMEDS)

                good_old = good_old[mask.ravel() == 1]
                good_new = good_new[mask.ravel() == 1]

                # print(good_new.shape)

                E = np.matmul(K_mat.T,(np.matmul(F,K_mat)))

                _, R, t, _ = cv.recoverPose(E, good_old, good_new,K_mat)

                R = R.T
                t = -np.matmul(R, t)

                if num_called == 1:
                    curr_R = R
                    pose = t
                else:
                    
                    curr_R = R.dot(prev_R)
                    pose = prev_R.dot(t) + prev_t

                prev_R = curr_R
                prev_t = pose

                trajectory.append(pose)

                print('Number of Iterations: ', num_called+1)
                print(pose)

                traj = np.asarray(trajectory).reshape(-1,3)
                traj = traj.T

                plt.plot(traj[0,:],traj[2,:])

                if len(files) == 0:
                    plt.pause(0)
                else:
                    plt.pause(0.1)

                plt.show()
                
                np.save('traj',trajectory)

if __name__ == "__main__":
    
    path = '/home/yagnesh/Desktop/GitHub-Repos/VisualOdom/dataset'
    scale_xy = 0.55

    start_tracking(path, scale_xy)