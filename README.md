# VisualOdom

## Monocular Visual Odometry
The method of determining motion of a monocular camera using sequential images is known as Monocular Visual Odometry.

### Steps involved in Visual Odometry:
1. Feature Detection: There are many different feature detection algorithms, most widely used ones include: SIFT, ORB, FAST, BRIEF and etc. The one being used in [optical_2d.py](https://github.com/yagdev99/VisualOdom/blob/main/optical_2d.py) is SIFT.
2. Matching features from previous image to next image.
3. Finding the relative rotation and translation from previous image to current image.
4. Finding the rotation and translation from the initial image.

Scale Invarient Feature Transform (SIFT) is very robust feature detection algorithm but it takes very long to to detect the features in an image. Additionally it taking time to match it with the features in the previous image. In order to optimize this we use Optical Flow Feature Tracking.  


## Result

![result](https://user-images.githubusercontent.com/69981745/125235520-385c2480-e300-11eb-9bbf-d7bb9e1eb8ca.png)

## Resources:
1. [Optical Flow](https://nanonets.com/blog/optical-flow/)
2. [Scaramuzza Part I](http://rpg.ifi.uzh.ch/docs/VO_Part_I_Scaramuzza.pdf)
3. [Scaramuzza Part II](https://www.zora.uzh.ch/id/eprint/71030/1/Fraundorfer_Scaramuzza_Visual_odometry.pdf)
