import os 
import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import measurements, morphology
from tqdm import tqdm
import glob
import json
import math
from scipy.spatial.distance import euclidean



def get_line_angle(line, clockwise):
        vector = line[1] - line[0]
        length = np.linalg.norm(vector)
        cos    = np.dot(vector, [0, 1]) / length
        angle = math.acos(cos) *180 / math.pi
        return angle if clockwise == True else -angle
def get_tooth_degree(points):
        points = sorted(points ,key=lambda point:point[1])
        up_center_point   = [(points[0][0]+points[1][0])/2,(points[0][1]+points[1][1])/2]
        down_center_point = [(points[2][0]+points[3][0])/2,(points[2][1]+points[3][1])/2]
        midline = [up_center_point,down_center_point]
        midline = np.array(midline)
        midline , midline_direction = recognize_line(midline)
        midline_angle = get_line_angle(midline, midline_direction)
        
#         clock = lambda x: "clockwise" if x == True else "counterclockwise"
        return midline_angle



def get_new_rotate_center(points, rotate_center):
        points = sorted(points ,key=lambda point:point[1])
        ux, uy = [(points[0][0]+points[1][0])/2,(points[0][1]+points[1][1])/2]
        dx, dy = [(points[2][0]+points[3][0])/2,(points[2][1]+points[3][1])/2]
        rcx, rcy = rotate_center
        
        t = (rcy-uy) / (dy-uy)
        new_rcx = ux + ( dx - ux ) * t
        return (int(new_rcx), rcy) 
def rotate(image, angle, image_center, max_bound_rect=True):
        
        height, width = image.shape[:2] # image shape has 3 dimensions
        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)
        
        if not max_bound_rect:
                rotated_image = cv2.warpAffine(image, rotation_mat, (width, height))
                return rotated_image
            
        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0,0]) 
        abs_sin = abs(rotation_mat[0,1])

        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        rotation_mat[0, 2] += bound_w/2 - image_center[0]
        rotation_mat[1, 2] += bound_h/2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix
        rotated_image = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))
        return rotated_image

def recognize_line(line):
        sort_up_and_down = lambda line: [line[1], line[0]] if line[1][1] < line[0][1] else line
        check_clockwise  = lambda line: True if line[0][0] > line[1][0] else False    # up >>>> down
        line = sort_up_and_down(line)
        clockwise = check_clockwise(line)
        return line, clockwise
    
class Dental_Rotation:
        
        @staticmethod
        def rotate(image, points, image_center, rotate_angle, max_rect=True):
                rotate_image = rotate(image, rotate_angle, image_center, max_rect)
                return rotate_image
            
        @staticmethod
        def centralize(image, points, max_rect=True):
                rotate_angle = get_tooth_degree(image, points)
                rotate_image = rotate(image, rotate_angle, max_rect)
                return rotate_image
            
class border_preprocessing:
       
        @staticmethod
        def trim_border(image):
                if not np.sum(image[0]):     
                        return border_preprocessing.trim_border(image[1:])

                elif not np.sum(image[-1]):  
                        return border_preprocessing.trim_border(image[:-2])

                elif not np.sum(image[:,0]):
                        return border_preprocessing.trim_border(image[:,1:]) 

                elif not np.sum(image[:,-1]):
                        return border_preprocessing.trim_border(image[:,:-2])    
                return image
        
        @staticmethod
        def padding(image, padding_height = 700, padding_width = 400):
                mask_size = (padding_height, padding_width)
                tooth_h, tooth_w = image.shape
                mask = np.zeros(mask_size)
                yoff = round((mask_size[0]-tooth_h)/2)
                xoff = round((mask_size[1]-tooth_w)/2)
                result = mask.copy()
                result[yoff:yoff+tooth_h, xoff:xoff+tooth_w] = image
                return result

def crop_image(tooth_img, point):
    points   = np.array(point).astype(int)
    points   = np.where(points < 0, 0, points)
#     tooth_img = cv2.imread(path.replace('result','image'),0)

    rect = cv2.boundingRect(points)
    x, y, w, h = rect

    zero_degree = int(get_tooth_degree(points))

    alpha = 0.66
    delta_w = abs(w * math.sin( zero_degree * math.pi  / 180 ) * alpha)

    delta_degree = abs(zero_degree) - 12
    delta_extra = 0
    if delta_degree > 0 :
            delta_extra = delta_degree ** 1.2 / 200 * w # 200 more or less

    x = int(x - 0.5 * (delta_w + delta_degree) )
    w = int(w - delta_w - delta_degree )

    rotate_center = ( (2*x+w)//2, (2*y+h)//2 )
    new_rotate_center = get_new_rotate_center(points, rotate_center)
    delta_x = new_rotate_center[0] -  rotate_center[0]
    rotate_center = new_rotate_center

    if x + delta_x < 0: 
            x = 0
            w = w + x + delta_x
    else: 
            x = x + delta_x

    if x + w + delta_x > tooth_img.shape[1]: 
            w = x + 2 * w + delta_x - tooth_img.shape[1]


    deg_lowbound, deg_upperbound, step = (0, 1, 1)#self.rotate_range
    deg_lowbound -= zero_degree
    deg_upperbound -= zero_degree
    
    for idx, degree in enumerate(range(deg_lowbound, deg_upperbound, step), 1):
        rect = cv2.boundingRect(points)

        delta_w2 = int(0.33 * h * math.sin( abs(degree + zero_degree ) * math.pi / 180 ))
        mask = np.zeros(tooth_img.shape, np.uint8)
        mask[y:y+h, max(x-delta_w2, 0): min(x+delta_w2+w, tooth_img.shape[1]) ] = 255


        ro_mask = Dental_Rotation.rotate(mask, points, rotate_center, degree, max_rect=False)

        ro_tooth = cv2.bitwise_and(tooth_img, tooth_img, mask=ro_mask)
        ro_tooth = rotate(ro_tooth, -degree, rotate_center)
        ro_tooth = border_preprocessing.trim_border(ro_tooth)
       
#     plt.figure(figsize=(6,3))
#     plt.imshow(ro_tooth,cmap='gray')
#     plt.show()
    return ro_tooth, ro_mask

