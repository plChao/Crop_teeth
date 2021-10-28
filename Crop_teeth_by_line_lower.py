###
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
from Cropping_teeth_function import *
from datetime import datetime
start_time = datetime.now()
###
# img_path = 'D:/Lab/PBL/tooth_detection/unet/test/seg_image/*.PNG'
img_path = './choose_crop_img/lower/*.PNG'
o_image_folder_path = '../root_and_image_data/1_o_image/'
# o_image_folder_path = '../../data/root_and_image_data/1_o_image/'
img_path_list = glob.glob(img_path)
assert len(img_path_list) > 0


###
###############################cell
def get_tooth_degree(points):
    midline = np.array(points)
    midline , midline_direction = recognize_line(midline)
    midline_angle = get_line_angle(midline, midline_direction)
    return midline_angle

def get_inertia_parameter(img_array):
    try:
        y, x = np.nonzero(img_array)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        x = x - x_mean
        y = y - y_mean
        coords = np.vstack([x, y])
        cov = np.cov(coords)
        evals, evecs = np.linalg.eig(cov)
        sort_indices = np.argsort(evals)[::-1]
        x_v1, y_v1 = evecs[:, sort_indices[0]]
        x_v2, y_v2 = evecs[:, sort_indices[1]]
        return x_v1, y_v1, x_mean, y_mean,len(x)
    except:
        return 4,3,2,1,0

def inertia_detection_cv_line(img_path,draw_img=None,nparray=True,scale=100,width=5,ignore_slope=False):
    if nparray:
        img = img_path
    else:
        img = scipy.misc.imread(img_path, flatten=1)
    x_v1, y_v1, x_mean, y_mean,len_non_zero = get_inertia_parameter(img)
#     scale = 5
#     width = 0
    try:
        try:
            if draw_img.all() != None:
                img = draw_img
        except:
            pass
        if len_non_zero > 0 :
            if ignore_slope:
                cv2.line(img, (int(x_v1*-scale*2+x_mean),int(y_v1*-scale*2+y_mean)),(int(x_v1*scale*2+x_mean),int(y_v1*scale*2+y_mean)), (255, 0, 255), width)
                return img,[(int(x_v1*-scale*2+x_mean),int(y_v1*-scale*2+y_mean)),(int(x_v1*scale*2+x_mean),int(y_v1*scale*2+y_mean))]
            elif slope(-x_v1,-y_v1,x_v1,y_v1) > 1.2:
                cv2.line(img, (int(x_v1*-scale*2+x_mean),int(y_v1*-scale*2+y_mean)),(int(x_v1*scale*2+x_mean),int(y_v1*scale*2+y_mean)), (255, 0, 255), width)
                return img,[(int(x_v1*-scale*2+x_mean),int(y_v1*-scale*2+y_mean)),(int(x_v1*scale*2+x_mean),int(y_v1*scale*2+y_mean))]
            else:
                return img,[(0,0),(1,1)]
        else:
            return img,[(0,0),(1,1)]
    except Exception as e: 
        print('except')
        print(e)
        return img,[(0,0),(1,1)]
    
def get_width(img_array):
    nums = 60
    y, x = np.nonzero(img_array)
    sort_width_list = sorted([(y_,list(y).count(y_)) for y_ in set(y)], reverse=True ,key=lambda x:x[1])[:nums]
    width = int(sum(value[1] for value in sort_width_list)/nums)
    return width

def get_hight(img_array):
    y, x = np.nonzero(img_array)
    hight = max(y) - min(y)
    return hight
#########################

def IsBridge(mul_ro_tooth, auto_crop_temp):
    u, idx, inv = colorful_unique(mul_ro_tooth)
    if (0,255,0) in u.tolist():
        green_idx = u.tolist().index((0,255,0))
    else:
        green_idx = -1

    if (0,255,255) in u.tolist():
        light_blue_idx = u.tolist().index((0,255,255))
    else:
        light_blue_idx = -1

    crown = 0
    for color_num in range(len(inv)):
        if color_num in [green_idx, light_blue_idx]:
            crown+=inv[color_num]

    try:
        if crown/np.unique(auto_crop_temp, return_counts=True)[1][1]>0.5:
            return True
        else:
            return False
    except:
        return False


def crop_teeth_byline(path, o_image, final_mul,final_mul_only_one_line, upper_teeth,o_image_folder,mul_ori,ori_teeth_mask,result_img_ori):
    tmp = final_mul.copy()
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    condition = (tmp==29) | (tmp==0) | (tmp==76)
    tmp2 = np.where(~condition,255,0)
    
    grow = morphology.binary_dilation(tmp2, structure=np.ones((1, 1), dtype=int))
    lbl, npatches = measurements.label(grow)
    lbl[tmp2==0] = 0

    unique, counts = np.unique(lbl, return_counts=True)
    key_list = dict(zip(unique, counts))
    components_list = []
    for index, pixel in enumerate(key_list):
        if len(key_list)==1:
            img_gray = lbl.copy()
            if np.where(img_gray == pixel, 1, 0).sum()<1000:
                continue
            component = np.where(img_gray == pixel, 255, 0)
            components_list.append(component)
        elif index != 0 :
            img_gray = lbl.copy()
            if np.where(img_gray == pixel, 1, 0).sum()<1000:
                continue
            component = np.where(img_gray == pixel, 255, 0)
            components_list.append(component)
    
    x_mean_list = []
    ro_tooth_list = []
    t_ori_list = []
    mask_list = []
    box_list = []
    
    ro_tooth_list_ordered = []
    t_ori_list_ordered = []
    mask_list_ordered = []
    box_list_ordered = []
    
    bubble_ro_tooth_list_ordered = []
    bubble_t_ori_list_ordered = []
    bubble_mask_list_ordered = []
    bubble_box_list_ordered = []
    bubble_missing_teeth = []
    
    auto_crop_list = []
    auto_crop_list_ordered = []
    teeth_mask_list = []
    teeth_mask_list_ordered = []
    mul_ro_tooth_list = []
    mul_ro_tooth_list_ordered = []
    
    o_image_t = cv2.cvtColor(o_image, cv2.COLOR_GRAY2BGR)
    merge_o_image_t = o_image_t.copy()
    merge_mul_ori = mul_ori.copy()
    color_list = [(160,32,240),(0,199,140),(255,125,64),(64,224,208)] #(107,142,35)
    for i in range(len(components_list)):
        o_image_t = cv2.cvtColor(o_image, cv2.COLOR_GRAY2BGR)
        t_ori = o_image_t.copy()
        
        o_image_t = cv2.resize(o_image_t,(round(o_image_t.shape[1]*0.25),round(o_image_t.shape[0]*0.25)))
        lbl = components_list[i].copy().astype('uint8')
        lbl_copy = lbl.copy()
        
        points = cv2.findNonZero(lbl)
        rect = cv2.minAreaRect(points)

        hight = get_hight(lbl)
        width = get_width(lbl)
        inertia_image, line_of_inertia_1 = inertia_detection_cv_line(lbl,draw_img=None,nparray=True,scale=int(hight*0.35),width=int(width*0.35),ignore_slope=True)
        rotate_degree = get_tooth_degree(np.array(line_of_inertia_1))
        
        min_angle = rect[2]
        if rotate_degree>45:
            rotate_degree -= 90
            ref = (np.array(line_of_inertia_1[0])+np.array(line_of_inertia_1[1]))//2
            line_of_inertia_1 = (rotate_line(line_of_inertia_1[0], ref, -90),rotate_line(line_of_inertia_1[1], ref, -90))
        elif rotate_degree<-45:
            rotate_degree += 90
            ref = (np.array(line_of_inertia_1[0])+np.array(line_of_inertia_1[1]))//2
            line_of_inertia_1 = (rotate_line(line_of_inertia_1[0], ref, 90),rotate_line(line_of_inertia_1[1], ref, 90))
        
        x_v1, y_v1, x_mean, y_mean,len_non_zero = get_inertia_parameter(lbl)
        
        if upper_teeth:
            x_mean = int(x_v1*+0.035*hight*2+x_mean)
            y_mean = int(y_v1*+0.035*hight*2+y_mean)
        else:
            # print('lower')
            x_mean = int(x_v1*-0.035*hight*2+x_mean)
            y_mean = int(y_v1*-0.035*hight*2+y_mean) # 0.03
        
        rect = ((int(x_mean), int(y_mean)),tuple([int(x*1.15) for x in rect[1]]),rotate_degree) #1.1
    
        box = find_boundary(lbl_copy, line_of_inertia_1)
        box_findoverlap = box #為了找overlap 較嚴謹
        
        box = expand_rectangle(box, scale = 0.05) #0.05 當用0.05時準確率達到0.88 並且為了切牙而大一點
        
        lbl = cv2.cvtColor(lbl, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(t_ori, [box*4], 0, (0,255,0), 5)
        cv2.drawContours(merge_o_image_t, [box*4], 0, color_list[i%4], 5)
        cv2.drawContours(merge_mul_ori, [box*4], 0, color_list[i%4], 5)

#         o_image_t = cv2.resize(o_image,(round(o_image.shape[1]*0.25),round(o_image.shape[0]*0.25)))
        o_image_t = o_image.copy()
    
#         mul_temp = result_img_ori,copy()#merge_mul_ori.copy()
#         mul_temp = cv2.cvtColor(mul_temp, cv2.COLOR_BGR2GRAY)
        try:
            ro_tooth, auto_crop = crop_image(o_image_t, box*4)
            
#             mul_ro_tooth, auto_crop_mul = crop_image(mul_temp, box*4)
            
            overlap_ro_tooth, overlap_auto_crop = crop_image(o_image_t, box_findoverlap*4)
        except Exception as e: 
            print(e)
            print(os.path.basename(path)[:-4])
            # print('box =',box*4)
            continue
        
        if max(ro_tooth.shape)/min(ro_tooth.shape)>5:
            
            cv2.imwrite('skip/'+os.path.basename(path)[:-4]+'_'+str(i)+'.png', ro_tooth)
            continue
#########################################################
        mul_temp = result_img_ori.copy()#merge_mul_ori.copy()
        auto_crop_temp = np.where(auto_crop>=128,255,0).astype('uint8')
        mul_ro_tooth = cv2.bitwise_and(mul_temp, mul_temp, mask=auto_crop_temp)
        
        #IsBridge(mul_ro_tooth)
        x_mean_list.append(int(x_mean))
        ro_tooth_list.append(ro_tooth)
        t_ori_list.append(t_ori)
        mask_list.append(lbl_copy)       
        box_list.append((box*4).tolist())
        
        auto_crop_list.append(auto_crop_temp)
        mul_ro_tooth_list.append(mul_ro_tooth)
        

        teeth_mask = cv2.bitwise_and(overlap_auto_crop, overlap_auto_crop, mask=ori_teeth_mask)
        teeth_mask_list.append(teeth_mask)
#     print('box_list =',np.array(box_list)//4)
        
#######################################################
    for num, i in enumerate(np.argsort(x_mean_list)):
        teeth_mask_list_ordered.append(teeth_mask_list[i])
        auto_crop_list_ordered.append(auto_crop_list[i])
        mul_ro_tooth_list_ordered.append(mul_ro_tooth_list[i])
        
        ro_tooth_list_ordered.append(ro_tooth_list[i])
        t_ori_list_ordered.append(t_ori_list[i])
        mask_list_ordered.append(mask_list[i])
        box_list_ordered.append(box_list[i])
        
    missing_teeth = []
    for i in range(len(teeth_mask_list_ordered)-1):
        teeth_mask = cv2.bitwise_and(teeth_mask_list_ordered[i], teeth_mask_list_ordered[i+1])
        if len(np.unique(teeth_mask)) == 1:
#             print("missing")
            missing_teeth.append(1)
        else:
#             print("not missing")
            missing_teeth.append(0)

    missing_teeth.append(0) #last tooth
#     bubble_ro_tooth_list_ordered = []
#     bubble_t_ori_list_ordered = []
#     bubble_mask_list_ordered = []
#     bubble_box_list_ordered = []
#     bubble_missing_teeth = []
    for i in range(len(ro_tooth_list_ordered)):
        bubble_ro_tooth_list_ordered.append(ro_tooth_list_ordered[i])
        bubble_t_ori_list_ordered.append(t_ori_list_ordered[i])
        bubble_mask_list_ordered.append(mask_list_ordered[i])
        bubble_box_list_ordered.append(box_list_ordered[i])
        if IsBridge(mul_ro_tooth_list_ordered[i], auto_crop_list_ordered[i]):
            bubble_missing_teeth.append(2)
        else:
            bubble_missing_teeth.append(0)
        if missing_teeth[i]==1:
            mm = np.zeros((50,50),dtype='uint8')
            bubble_ro_tooth_list_ordered.append(mm)
            bubble_t_ori_list_ordered.append(mm)
            bubble_mask_list_ordered.append(mm)
            bubble_box_list_ordered.append(box_list_ordered[0])
            bubble_missing_teeth.append(1)
    
    write_missing_json_file(bubble_missing_teeth, o_image_folder, path)
    write_box_json_file(bubble_box_list_ordered, o_image_folder, path)
    for i in range(len(bubble_ro_tooth_list_ordered)):
        
        plt.figure(figsize=(20,10))
        plt.subplot(141)
        plt.imshow(bubble_t_ori_list_ordered[i])
        plt.subplot(142)
        plt.imshow(final_mul_only_one_line)
        plt.subplot(143)
        plt.imshow(bubble_mask_list_ordered[i],cmap='gray')
        plt.subplot(144)
        plt.imshow(bubble_ro_tooth_list_ordered[i],cmap='gray')
        plt.savefig('crop_teeth/'+os.path.basename(path)[:-4]+'_'+str(i)+'.png')
        plt.clf()
        plt.close('all')
        # plt.show()
        if not os.path.exists('folder_teeth/'+o_image_folder):
            os.mkdir('folder_teeth/'+o_image_folder)
        cv2.imwrite('folder_teeth/'+o_image_folder+'/'+os.path.basename(path)[:-4]+'_'+str(i)+'.png', bubble_ro_tooth_list_ordered[i])
        cv2.imwrite('teeth/'+os.path.basename(path)[:-4]+'_'+str(i)+'.png', bubble_ro_tooth_list_ordered[i])
        
        cv2.imwrite('mask/'+os.path.basename(path)[:-4]+'_'+str(i)+'.png', bubble_mask_list_ordered[i])
    
    
    if not os.path.exists('drawCountour_mul/'+o_image_folder):
        os.mkdir('drawCountour_mul/'+o_image_folder)
    plt.figure(figsize=(12,10))
    plt.subplot(121)
    plt.imshow(merge_o_image_t)
    plt.subplot(122)
    plt.imshow(merge_mul_ori)
    plt.savefig('drawCountour_mul/'+o_image_folder+'/'+os.path.basename(path))
#     plt.show()
    plt.clf()
    plt.close('all')

    
    if not os.path.exists('drawContour_folder/'+o_image_folder):
        os.mkdir('drawContour_folder/'+o_image_folder)
    cv2.imwrite('drawContour_folder/'+o_image_folder+'/'+os.path.basename(path), merge_o_image_t)
#     return bubble_ro_tooth_list_ordered

def write_missing_json_file(missing_teeth, o_image_folder, path):
    
    data = {}
    data['shapes'] = []
    for i in range(len(missing_teeth)):
        data['shapes'].append({
            'miss': missing_teeth[i],
            'name': os.path.basename(path)[:-4]+'_'+str(i)+'.png'
        })
    if not os.path.exists('missing_json_file/'+o_image_folder):
        os.mkdir('missing_json_file/'+o_image_folder)
    with open('missing_json_file/'+o_image_folder+'/'+os.path.basename(path)[:-4]+'.json', 'w') as outfile:
        json.dump(data, outfile)

def write_box_json_file(box_list_ordered, o_image_folder, path):
    
    data = {}
    data['shapes'] = []
    for i in range(len(box_list_ordered)):
        data['shapes'].append({
            'points': box_list_ordered[i],
            'name': os.path.basename(path)[:-4]+'_'+str(i)+'.png'
        })
    if not os.path.exists('box_json_file/'+o_image_folder):
        os.mkdir('box_json_file/'+o_image_folder)
    with open('box_json_file/'+o_image_folder+'/'+os.path.basename(path)[:-4]+'.json', 'w') as outfile:
        json.dump(data, outfile)

def find_boundary(lbl_copy, line_of_inertia_1):
    
    for i in range(lbl_copy.shape[1]):
        target_line = ((line_of_inertia_1[0][0]+i,line_of_inertia_1[0][1]),(line_of_inertia_1[1][0]+i,line_of_inertia_1[1][1]))
        mask = np.zeros((lbl_copy.shape[0],lbl_copy.shape[1]),dtype=np.uint8)
        cv2.line(mask, target_line[0], target_line[1], 255, 1)
        line = cv2.bitwise_and(lbl_copy, lbl_copy, mask=mask)
        if len(np.unique(line)) == 1:
            break
    line_right = target_line

    for i in range(lbl_copy.shape[1]):
        target_line = ((line_of_inertia_1[0][0]-i,line_of_inertia_1[0][1]),(line_of_inertia_1[1][0]-i,line_of_inertia_1[1][1]))
        mask = np.zeros((lbl_copy.shape[0],lbl_copy.shape[1]),dtype=np.uint8)
        cv2.line(mask, target_line[0], target_line[1], 255, 1)
        line = cv2.bitwise_and(lbl_copy, lbl_copy, mask=mask)
        if len(np.unique(line)) == 1:
            break
    line_left = target_line
    
    ref = (np.array(line_of_inertia_1[0])+np.array(line_of_inertia_1[1]))//2
    hor_inertia_1 = (rotate_line(line_of_inertia_1[0], ref, 90),rotate_line(line_of_inertia_1[1], ref, 90))
    for i in range(lbl_copy.shape[0]):
        target_line = ((hor_inertia_1[0][0],hor_inertia_1[0][1]-i),(hor_inertia_1[1][0],hor_inertia_1[1][1]-i))
        mask = np.zeros((lbl_copy.shape[0],lbl_copy.shape[1]),dtype=np.uint8)
        cv2.line(mask, target_line[0], target_line[1], 255, 1)
        line = cv2.bitwise_and(lbl_copy, lbl_copy, mask=mask)
        if len(np.unique(line)) == 1:
            break
    line_up = target_line

    for i in range(lbl_copy.shape[0]):
        target_line = ((hor_inertia_1[0][0],hor_inertia_1[0][1]+i),(hor_inertia_1[1][0],hor_inertia_1[1][1]+i))
        mask = np.zeros((lbl_copy.shape[0],lbl_copy.shape[1]),dtype=np.uint8)
        cv2.line(mask, target_line[0], target_line[1], 255, 1)
        line = cv2.bitwise_and(lbl_copy, lbl_copy, mask=mask)
        if len(np.unique(line)) == 1:
            break
    line_down = target_line

    point1 = line_intersection(line_down, line_left)
    point2 = line_intersection(line_left, line_up)
    point3 = line_intersection(line_up, line_right)
    point4 = line_intersection(line_right, line_down)

    return np.array([point1,point2,point3,point4])
    

def Is_upper_teeth(combined_masks):
    if 255 in (np.unique(combined_masks[-combined_masks.shape[1]//3:],return_counts=True)[0]):
        bot = np.unique(combined_masks[-combined_masks.shape[1]//3:],return_counts=True)[1][1]
    else:
        bot = 0
    if 255 in (np.unique(combined_masks[:combined_masks.shape[1]//3+1],return_counts=True)[0]):
        top = np.unique(combined_masks[:combined_masks.shape[1]//3+1],return_counts=True)[1][1]
    else:
        top = 0
    if bot>top:
        return False
    else:
        return True

def rotate_line(point, origin, degrees):
    radians = np.deg2rad(degrees)
    x,y = point
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = np.cos(radians)
    sin_rad = np.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
#     print(qx,qy)
    return round(qx), round(qy)

def colorful_unique(x):
    dt = np.dtype([('a', x.dtype), ('b', x.dtype), ('c', x.dtype)])
    y = x.view(dtype=dt).squeeze()
    u, idx, inv = np.unique(y, return_counts=True, return_inverse=True)
#     print(u)
    return u, idx, inv


def extend_line(s,e,img):
    s = np.array(s)
    e = np.array(e)
#     scale = 1
    vec = e-s
    
#     print(vec)
    if vec[0]!=0:
        if vec[1]!=0:
            scale = max(abs(img.shape[0]/vec[0]),abs(img.shape[1]/vec[1]))
        else:
            scale = abs(img.shape[0]/vec[0])
    else:
        if vec[1]==0:
            scale = 10
        else:
            scale = abs(img.shape[1]/vec[1])
    
    scale = round(scale)+1
    e = e + scale*vec
    s = s - scale*vec
#     print(e)
    return tuple(s), tuple(e)
def restrict_extend_line(s, e, img, w, b):
    s = np.array(s)
    e = np.array(e)
    x = round(s[0]-((s[0]-e[0])/(s[1]-e[1]))*(s[1]-w))
    s = np.array((x,w))
    x = round(e[0]-((s[0]-e[0])/(s[1]-e[1]))*(e[1]-b))
    e = np.array((x,b))
    return tuple(s), tuple(e)
    
    

def expand_rectangle(rec, scale = 0.05):
    point1 = rec[0] + (rec[0]- rec[1] + rec[0] - rec[3])*scale
    point2 = rec[1] + (rec[1]- rec[0] + rec[1] - rec[2])*scale
    point3 = rec[2] + (rec[2]- rec[3] + rec[2] - rec[1])*scale
    point4 = rec[3] + (rec[3]- rec[0] + rec[3] - rec[2])*scale
    return np.array([point1,point2,point3,point4],dtype=int)

def find_max_nonzeros_component(components_list):
    find_max_idx = -1
    max_nonzero = -1
    for i in range(len(components_list)):
        if len(np.nonzero(components_list[i])[0]) > max_nonzero:
            max_nonzero = len(np.nonzero(components_list[i])[0])
            find_max_idx = i
    return find_max_idx

def remove_hole(gray):
    im_floodfill = gray.copy()
    im_th = gray.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = im_th | im_floodfill_inv

    return im_out

def remove_small_component(result_img):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(result_img, connectivity=8)

    sizes = stats[1:, -1]; nb_components = nb_components - 1

    min_size = 1000  

    #your answer image
    img2 = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    img2 = img2.astype('uint8')

    img2[0,:] = 0
    img2[img2.shape[0]-1,:] = 0
    img2[:,0] = 0
    img2[:,img2.shape[1]-1] = 0
    
    
    img2 = remove_hole(img2)

#     result_img = img2.copy()
    return img2

def find_black_component(result_img_gray):
    condition = (result_img_gray==0)
    result_img_gray = np.where(condition,255,0).astype('uint8')
    result_img_gray[0,:] = 255
    result_img_gray[result_img_gray.shape[0]-1,:] = 255
    result_img_gray[:,0] = 255
    result_img_gray[:,result_img_gray.shape[1]-1] = 255
    
    grow = morphology.binary_dilation(result_img_gray, structure=np.ones((1, 1), dtype=int))
    lbl, npatches = measurements.label(grow)
    lbl[result_img_gray==0] = 0
    
    unique, counts = np.unique(lbl, return_counts=True)
    key_list = dict(zip(unique, counts))
    components_list = []
    for index, pixel in enumerate(key_list):
        if len(key_list)==1:
            img_gray = lbl.copy()
            component = np.where(img_gray == pixel, 255, 0)
            components_list.append(component)
        elif index != 0 :
            img_gray = lbl.copy()
            component = np.where(img_gray == pixel, 255, 0)
            unique, counts = np.unique(component, return_counts=True)
#             if counts[1] < 10:
#                 continue
            
            components_list.append(component)
    
    combined_masks = np.zeros((result_img_gray.shape[0],result_img_gray.shape[1]),dtype=np.uint8)
    for i in range(1,len(components_list)):
        combined_masks[components_list[i]==255]=255
    
    return components_list[0], combined_masks

def point_center(component):
    background = component.copy().astype('uint8')
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
    
    gray = np.float32(component)
#     dst = cv2.cornerHarris(gray,3,3,0.04)
    dst = cv2.cornerHarris(gray,5,5,0.04)
    dst = cv2.dilate(dst,None)
    background[dst>0.1*dst.max()]=[255,0,0]
    
    background = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
    temp = np.where(background==76,255,0).astype('uint8')
    
    
    grow = morphology.binary_dilation(temp, structure=np.ones((1, 1), dtype=int))
    lbl, npatches = measurements.label(grow)
    lbl[result_img_gray==0] = 0

    unique, counts = np.unique(lbl, return_counts=True)
    key_list = dict(zip(unique, counts))
    point_list_red = []
    for index, pixel in enumerate(key_list):
        if len(key_list)==1:
            img_gray = lbl.copy()
            component = np.where(img_gray == pixel, 255, 0)
            center = (round(np.nonzero(component)[1].mean()),round(np.nonzero(component)[0].mean()))
            point_list_red.append(center)
        elif index != 0 :
            img_gray = lbl.copy()
            component = np.where(img_gray == pixel, 255, 0)
            unique, counts = np.unique(component, return_counts=True)
            center = (round(np.nonzero(component)[1].mean()),round(np.nonzero(component)[0].mean()))
            point_list_red.append(center)
    return point_list_red

def cal_ref_point(i,img_shape,point_component):
    min_dst = 99999
    for back_point in point_component[0]:
        if 0<=back_point[0]<=5 or 0<=back_point[1]<=5 or img_shape[0]-6<=back_point[0]<=img_shape[0]-1 or img_shape[1]-6<=back_point[1]<=img_shape[1]-1:
            continue
        for teeth_point in point_component[i]:
            dst = euclidean(teeth_point, back_point)
            if min_dst > dst:
                min_dst = dst
                back_ref_point = back_point
                teeth_ref_point = teeth_point
    return back_ref_point,  teeth_ref_point

def check_line_color(back_ref_point,teeth_ref_point,result_img_gray_ori):
    check = [105,150,179,0]
    for i in range(5):
        x = round(back_ref_point[0]+(teeth_ref_point[0]-back_ref_point[0])*0.2*i)
        y = round(back_ref_point[1]+(teeth_ref_point[1]-back_ref_point[1])*0.2*i)
        if result_img_gray_ori[y,x] not in check:
            print(result_img_gray_ori[y,x])
            return False
    return True

def pixel_color_in_line(img, origin, ref_point, angle = 0):
    
    NewPointA = rotate_line(origin,ref_point,angle)
    extend_e, extend_s = extend_line(ref_point, NewPointA, img)

    mask = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
    cv2.line(mask, extend_e, extend_s, 255, 1)

    _, counts=np.unique(mask,return_counts=True)
    total_line_pixel = counts[1]

    line = cv2.bitwise_and(result_img, result_img, mask=mask)
    u, idx, inv = colorful_unique(line)
    return u, idx, inv

def find_pixel_in_line(img, non_background_component, origin, ref_point, angle = 0, w = 0, b = 0, draw=False):
#     print('origin,ref_point =',(origin,w,b))
    # w b = 15 145 
#     origin = (origin[0],)
    origin = (origin[0],b)
    extend_e = rotate_line(origin,ref_point,angle)
    origin = (origin[0],w)
    extend_s = rotate_line(origin,ref_point,angle)
#     NewPointA = rotate_line(origin,ref_point,angle)
#     extend_e, extend_s =  ref_point, NewPointA
    
#     extend_e, extend_s = extend_line(ref_point, NewPointA, img)
    
#     extend_e, extend_s = restrict_extend_line(extend_e, extend_s, img, w, b)

    mask = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
    cv2.line(mask, extend_e, extend_s, 255, 1)

    mask = cv2.bitwise_and(mask, non_background_component)
    
    _, counts=np.unique(mask,return_counts=True)
    
    if len(counts) == 1:
        total_line_pixel = 1
    else:
        total_line_pixel = counts[1]

    line = cv2.bitwise_and(result_img, result_img, mask=mask)
#     plt.imshow(line, cmap='gray')
#     plt.show()
    u, idx, inv = colorful_unique(line)

    if (255,255,0) in u.tolist():
        yellow_idx = u.tolist().index((255,255,0))
    else:
        yellow_idx = -1

    if (0,255,0) in u.tolist():
        green_idx = u.tolist().index((0,255,0))
    else:
        green_idx = -1

    if (0,255,255) in u.tolist():
        light_blue_idx = u.tolist().index((0,255,255))
    else:
        light_blue_idx = -1
        
    if (125,0,125) in u.tolist():
        pink_idx = u.tolist().index((125,0,125))
    else:
        pink_idx = -1

    other_pixel = 0
    for color_num in range(len(inv)):
        if color_num in [yellow_idx, green_idx, light_blue_idx,pink_idx]:
            other_pixel+=inv[color_num]
    
    if draw:
        temp = img.copy()
        cv2.line(temp, extend_e, extend_s, 255, 1)
        cv2.circle(temp, ref_point, 3, (255,0,0), 1)
        return extend_e, extend_s
        
    return other_pixel, total_line_pixel-other_pixel, other_pixel/total_line_pixel

def find_y_gap(result_img):
    wid = 0
    bid = result_img.shape[0]-1
    wid_tmp = 0
    bid_tmp = result_img.shape[0]-1
    if (125,0,125) in colorful_unique(result_img[0])[0].tolist() or (0,255,255) in colorful_unique(result_img[0])[0].tolist() or (0,255,0) in colorful_unique(result_img[0])[0].tolist() or (255,255,0) in colorful_unique(result_img[0])[0].tolist():
        last_flag = True
    else:
        last_flag = False
    flag = last_flag    
    max_gap = 0
    for w in range(1,result_img.shape[0]):
        if w == result_img.shape[0]-1:
            bid_tmp = w
            if max_gap < bid_tmp-wid_tmp and wid_tmp!=result_img.shape[0]:
                wid = wid_tmp
                bid = bid_tmp
            break
        if (125,0,125) in colorful_unique(result_img[w])[0].tolist() or (0,255,255) in colorful_unique(result_img[w])[0].tolist() or (0,255,0) in colorful_unique(result_img[w])[0].tolist() or (255,255,0) in colorful_unique(result_img[w])[0].tolist():
            flag = True
        else:
            flag = False
        if last_flag==False and flag == True:
            wid_tmp = w-1
        elif last_flag==True and flag == False:
            bid_tmp = w
            if max_gap < bid_tmp-wid_tmp:
                wid = wid_tmp
                bid = bid_tmp
                max_gap = bid_tmp-wid_tmp
            wid_tmp = result_img.shape[0]
        last_flag = flag
    return wid, bid
def extend_to_border(p1, p2, shape):
    # 2021/10/28 add by plchao, if work please write note
    x1, y1 = p1
    x2, y2 = p2
    boardx, boardy, _ = shape
    boardy = max(boardx, boardy)
    slope =  (y1 - y2) / (x1 - x2)
    assert type(slope) == float
    # x/y
    new_start_x, new_start_y = (boardy - y1) / slope + x1, boardy
    new_end_x, new_end_y = (0 - y1)/slope + x1, 0
    return (round(new_end_x), new_end_y), (round(new_start_x), new_start_y)

def choose_line_draw(tmp, result_img, non_background_component, init_angle):
    moving_range = 10
    angle_range = 30 + moving_range # 60
    x_axis = [i*0.5 - init_angle for i in range(-angle_range,angle_range+1)]
    x_axis = x_axis[moving_range:-moving_range]
    m_img = result_img.copy()
    line_list = []
    
    wid, bid = find_y_gap(result_img)
    
    for i in range(0,len(tmp)):
#         print(tmp[i])

        point_center = tmp[i]
        ref_point = point_center
        origin = ( point_center[0], result_img.shape[0])

        
        img = result_img.copy()

        other_pixel_list = []
        blue_black_color_list = []
        ratio_list = []
        moving_average_ratio_list = []

        for j in range(-angle_range,angle_range+1):
            xn = j*0.5 - init_angle
            other_pixel, blue_black_color, ratio = find_pixel_in_line(img, non_background_component, origin, ref_point, xn, wid, bid, draw=False)
            other_pixel_list.append(other_pixel)
            blue_black_color_list.append(blue_black_color)
            ratio_list.append(ratio)


        for num in range(moving_range,len(ratio_list)-moving_range):
            moving_average_ratio_list.append(sum(ratio_list[num-moving_range:num+moving_range+1]))

        other_pixel_list = other_pixel_list[moving_range:-moving_range]
        blue_black_color_list = blue_black_color_list[moving_range:-moving_range]
        ratio_list = ratio_list[moving_range:-moving_range]      
        extend_e, extend_s = find_pixel_in_line(img, non_background_component, origin, ref_point, x_axis[np.argmin(moving_average_ratio_list)], wid, bid, draw=True)
#         extend_e, extend_s = find_pixel_in_line(img, non_background_component, origin, ref_point, -init_angle, wid, bid,draw=True)

        u, idx, inv = pixel_color_in_line(img, extend_e, extend_s, angle = 0)

#########################################################################################
        
        if (125,0,125) in u.tolist(): # check if multi-root
            continue
            
        cv2.circle(m_img, ref_point, 1, (255,0,0), 1)
        extend_e, extend_s = extend_to_border(extend_e, extend_s, m_img.shape)
        cv2.line(m_img, extend_e, extend_s, (255,0,0), 1)
        line_list.append((tuple(np.array(extend_e)*4), tuple(np.array(extend_s)*4)))
#########################################################################################
        
    # plt.figure(figsize=(8,4))
    # plt.imshow(m_img)
    # plt.show()
    return m_img, line_list

def find_minrec(img, img_box):
    mult = 1.2
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    long_length_num = 0
    group0 = False
    boxes = []
    unsort_boxes = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
    #     cv2.drawContours(img_box, [box], 0, (0,255,0), 2) # this was mostly for debugging you may omit

        W = rect[1][0]
        H = rect[1][1]

        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)

        rotated = False
        angle = rect[2]

        if angle < -45:
            angle+=90
            rotated = True

        center = (int((x1+x2)/2), int((y1+y2)/2))
        size = (int(mult*(x2-x1)),int(mult*(y2-y1)))
    #     cv2.circle(img_box, center, 10, (0,255,0), -1) #again this was mostly for debugging purposes

        M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)

        cropped = cv2.getRectSubPix(img_box, size, center) 

        if cropped is None:
            continue
        cropped = cv2.warpAffine(cropped, M, size)

        croppedW = W if not rotated else H 
        croppedH = H if not rotated else W

        croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW*mult), int(croppedH*mult)), (size[0]/2, size[1]/2))

        x = np.where(croppedRotated>128,1,0)
        if np.sum(x) < 1000:
            continue

        x = np.where(croppedRotated>240,255,0).astype('uint8')
        x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, (5,5))

        cv2.drawContours(img_box, [box], 0, (0,255,0), 2) # this was mostly for debugging you may omit
        cv2.circle(img_box, center, 10, (0,255,0), -1) #again this was mostly for debugging purposes

        W = int(croppedW*mult)
        H = int(croppedH*mult)
#         print('ratio :',H/W)
        if H/W > 3:
            long_length_num += 1
#             cv2.drawContours(temp_img_ori, [box], 0, (0,255,0), 2)
#             cv2.circle(temp_img_ori, center, 10, (0,255,0), -1)
        unsort_boxes.append(box)
        box = box[box[:,1].argsort(kind='mergesort')]
        boxes.append(box)
#         plt.imshow(croppedRotated)
#         plt.show()
    if long_length_num == len(contours):
        group0 = True
    return long_length_num, group0, boxes, unsort_boxes

def get_pulp_img(mul_ori):
    image = mul_ori.copy()
    black_pixels_mask = np.all(image == [255, 0, 0], axis=-1)
    non_black_pixels_mask = np.any(image != [255, 0, 0], axis=-1)  
    image[black_pixels_mask] = [255, 255, 255]
    image[non_black_pixels_mask] = [0, 0, 0]
    return image

def combine_pulp(path,mul_image):
    ori_mul = mul_image.copy()
    kernel = np.ones((10,10),np.uint8)
    kernel_ero = np.ones((60,15), np.uint8)
    kernel_dila = np.ones((70,15), np.uint8)
    
    result_img = get_pulp_img(mul_image)
    result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2GRAY)
    result_img = np.where(result_img>128,255,0).astype('uint8')
    
    result_img = cv2.morphologyEx(result_img, cv2.MORPH_CLOSE, kernel)
    result_img = cv2.dilate(result_img, kernel_dila, iterations = 1)
    result_img = cv2.erode(result_img, kernel_ero, iterations = 1)

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(result_img, connectivity=8)

    sizes = stats[1:, -1]; nb_components = nb_components - 1

    min_size = 500  

    #your answer image
    img2 = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    img2 = img2.astype('uint8')
    img2[0,:] = 0
    img2[img2.shape[0]-1,:] = 0
    img2[:,0] = 0
    img2[:,img2.shape[1]-1] = 0
    img2 = remove_hole(img2)
    
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            if img2[i][j] == 255:
                mul_image[i][j] = (125,0,125)
            elif (mul_image[i][j]==(255,0,0)).all():
                mul_image[i][j] = (255,255,0)
    return mul_image


def crop_teeth_byline_show(path, o_image, final_mul,final_mul_only_one_line, mul_ori):
    tmp = final_mul.copy()
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    condition = (tmp==29) | (tmp==0) | (tmp==76)
    tmp2 = np.where(~condition,255,0)
    
    grow = morphology.binary_dilation(tmp2, structure=np.ones((1, 1), dtype=int))
    lbl, npatches = measurements.label(grow)
    lbl[tmp2==0] = 0

    unique, counts = np.unique(lbl, return_counts=True)
    key_list = dict(zip(unique, counts))
    components_list = []
    for index, pixel in enumerate(key_list):
        if len(key_list)==1:
            img_gray = lbl.copy()
            if np.where(img_gray == pixel, 1, 0).sum()<1000:
                continue
            component = np.where(img_gray == pixel, 255, 0)
            components_list.append(component)
        elif index != 0 :
            img_gray = lbl.copy()
            if np.where(img_gray == pixel, 1, 0).sum()<1000:
                continue
            component = np.where(img_gray == pixel, 255, 0)
            components_list.append(component)
    
    t_ori_list = []
    o_image_t = cv2.cvtColor(o_image, cv2.COLOR_GRAY2BGR)
    t_ori = o_image_t.copy()
    t_ori_mim = o_image_t.copy()
    t_ori_merge = o_image_t.copy()
    o_image_t = cv2.resize(o_image_t,(round(o_image_t.shape[1]*0.25),round(o_image_t.shape[0]*0.25)))
    color_list = [(255,0,0),(0,255,0),(255,0,255),(0,255,128)]
    for i in range(len(components_list)):
#         plt.imshow(components_list[i])
#         plt.show()
        lbl = components_list[i].copy().astype('uint8')
        lbl_copy = lbl.copy()
#         return 
        points = cv2.findNonZero(lbl)
        rect = cv2.minAreaRect(points)
        
        hight = get_hight(lbl)
        width = get_width(lbl)
        inertia_image, line_of_inertia_1 = inertia_detection_cv_line(lbl,draw_img=None,nparray=True,scale=int(hight*0.35),width=int(width*0.35),ignore_slope=True)
        rotate_degree = get_tooth_degree(np.array(line_of_inertia_1))
        print('rotate_degree =',rotate_degree)
        min_angle = rect[2]
        if rotate_degree-min_angle>45:
            rotate_degree -= 90
        elif rotate_degree-min_angle<-45:
            rotate_degree += 90
        x_v1, y_v1, x_mean, y_mean,len_non_zero = get_inertia_parameter(lbl)
        #########################
#         print(rect[0],rect[1],rect[2])
        befor_rect = (rect[0],tuple([1.1*x for x in rect[1]]),rect[2])
        rect = ((int(x_mean), int(y_mean)),tuple([x*2 for x in rect[1]]),rotate_degree)
#         rect = (rect[0],tuple([1.1*x for x in rect[1]]),rect[2])
        #########################
#         print(lbl_copy.shape)
#         plt.imshow(lbl_copy,cmap='gray')
#         plt.show()
        center = (int(x_mean), int(y_mean))
        size = tuple([int(x*1) for x in rect[1]])
        angle = rotate_degree
    
        rect = ((int(x_mean), int(y_mean)),tuple([int(x*1.1/2) for x in rect[1]]),rotate_degree)
        
        box = cv2.boxPoints(rect)
        box = np.int0(box)
#         return box
        lbl = cv2.cvtColor(lbl, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(t_ori, [box*4], 0, color_list[i%4], 5)
#         cv2.circle(t_ori, (int(rect[0][0]*4),int(rect[0][1]*4)), 10, (255,0,0), -1)
    
#         box_ori = box
        box = box*4
        box = box[box[:,1].argsort(kind='mergesort')]
        p1 = tuple((box[0]+box[1])//2)
        p2 = tuple((box[2]+box[3])//2)
        cv2.line(t_ori,p1,p2,(0,125,255),5)
        cv2.circle(t_ori, (int(x_mean*4), int(y_mean*4)), 10, (255,0,0), -1)
        cv2.line(mul_ori,p1,p2,(0,125,255),5)
        
        
#####################################################
        merge_angle = mask_left_right(p1[0]//4,p1[1]//4,p2[0]//4,p2[1]//4,lbl_copy)    
        
        if merge_angle-min_angle>45:
            merge_angle -= 90
        elif merge_angle-min_angle<-45:
            merge_angle += 90
        
        rect = ((int(x_mean), int(y_mean)),tuple([int(x) for x in befor_rect[1]]),merge_angle)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(t_ori_merge, [box*4], 0, color_list[i%4], 5)
        cv2.circle(t_ori_merge, (int(x_mean*4), int(y_mean*4)), 10, (255,0,0), -1)

    
################################################
        rect = befor_rect
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(t_ori_mim, [box*4], 0, color_list[i%4], 5)
#         cv2.circle(t_ori, (int(rect[0][0]*4),int(rect[0][1]*4)), 10, (255,0,0), -1)
#################################################
    
    
#         cv2.drawContours(t_ori, [box*4], 0, (0,255,0), 5)
    t_ori_list.append(t_ori)

    o_image_t = o_image.copy()
    plt.figure(figsize=(20,10))
    plt.subplot(141)
    plt.imshow(t_ori_mim)
    plt.subplot(142)
    plt.imshow(t_ori)
    plt.subplot(143)
    plt.imshow(t_ori_merge)
    plt.subplot(144)
    plt.imshow(mul_ori)
#     plt.show()
#     return box*4
#     plt.savefig('crop_teeth/'+os.path.basename(path)[:-4]+'.png')
#     plt.clf()
#     plt.close('all')
    return components_list
        
def point_left_right(combined_masks,select_tmp,upper_teeth):
    remove_list = []
    for select_idx in range(len(select_tmp)):
#         select_idx = 2
        imgray = combined_masks.copy()
        
        contours = test_function(imgray)
        im = cv2.cvtColor(imgray,cv2.COLOR_GRAY2BGR)
        min_distance = 9999
        idx = -1
        for cnt in range(len(contours)):
            if min_distance > euclidean(select_tmp[select_idx],contours[cnt][0]):
                min_distance = euclidean(select_tmp[select_idx],contours[cnt][0])
                point = contours[cnt][0]
                idx = cnt
        
        
#         for i in range(-20,21):
#             print(i, contours[idx+i][0])
        
        cv2.circle(im, tuple(point), 10, (255,0,0), 3)
        cur = point
        center_point = point
        
        idx_range = []
#         step = 10 # 5
        for i in range(1,len(contours)):
            idx_range.append((idx+i)%len(contours))
        
        for i in range(1,len(idx_range)):
            if contours[idx_range[i]][0][1]<cur[1] and contours[idx_range[-i]][0][1]<cur[1]:
                sign = 1 # 1
                break
            elif contours[idx_range[i]][0][1]>cur[1] and contours[idx_range[-i]][0][1]>cur[1]:
                sign = -1
                break
            
        for cnt in idx_range[::2]:
            cnt = (cnt + 2)%len(contours)
            if sign*contours[cnt][0][1]<=sign*cur[1]:
                cur = contours[cnt][0]
                if cur[0]<=25 or cur[0]>=im.shape[1]-25:
                    break
            else:
                break
        cv2.circle(im, tuple(cur), 10, (0,255,255), 3)
        left_point = cur

        cur = point
        for cnt in idx_range[::-1][::2]:
            cnt = (cnt - 2)%len(contours)
            if sign*contours[cnt][0][1]<=sign*cur[1]:
                cur = contours[cnt][0]
                if cur[0]<=25 or cur[0]>=im.shape[1]-25:
                    break
            else:
                break
        cv2.circle(im, tuple(cur), 10, (0,255,0), 3)
        right_point = cur
        plt.imshow(im)
        plt.show()
        
        thresh = max(abs(center_point[1]-left_point[1]),abs(center_point[1]-right_point[1]))
        if thresh < combined_masks.shape[0]/20:
            remove_list.append(select_tmp[select_idx])
        elif upper_teeth:
            if abs(center_point[1]-left_point[1]) == thresh:
                if center_point[1]-left_point[1]<0:
                    remove_list.append(select_tmp[select_idx])
            elif center_point[1]-right_point[1]<0:
                remove_list.append(select_tmp[select_idx])
        else:
            if abs(center_point[1]-left_point[1]) == thresh:
                if center_point[1]-left_point[1]>0:
                    remove_list.append(select_tmp[select_idx])
            elif center_point[1]-right_point[1]>0:
                remove_list.append(select_tmp[select_idx])
            

    for remove_data in remove_list:
        select_tmp.remove(remove_data)
    return select_tmp,remove_list
#         break

def test_function(result_img_gray):
    imgray = result_img_gray.copy()

    imgray[0:2,:] = 0
    imgray[result_img_gray.shape[0]-1:result_img_gray.shape[0]+1,:] = 0
    imgray[:,0:2] = 0
    imgray[:,result_img_gray.shape[1]-3:result_img_gray.shape[1]-1] = 0

    im = cv2.cvtColor(imgray,cv2.COLOR_GRAY2BGR)

    ret,thresh = cv2.threshold(imgray,127,255,0)
    contours_all, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours_all

def judge_left_right_on_line(x1, y1,x2, y2,x, y):
    if (x-x1)*(y2-y1)-(y-y1)*(x2-x1) < 0:
        return False
    elif (x-x1)*(y2-y1)-(y-y1)*(x2-x1) > 0:
        return True
    
def mask_left_right(x1,y1,x2,y2,lbl_copy):
    ori_lbl = lbl_copy.copy()
    mask = np.zeros((lbl_copy.shape[0],lbl_copy.shape[1]),dtype=np.uint8)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if judge_left_right_on_line(x1, y1,x2, y2,j, i):
                mask[i][j:] = 255
                break
    right_mask = cv2.bitwise_and(ori_lbl, ori_lbl, mask=mask)
    mask = np.where(mask==255,0,255).astype('uint8')
    left_mask = cv2.bitwise_and(ori_lbl, ori_lbl, mask=mask)
    
    right_img = cv2.cvtColor(right_mask, cv2.COLOR_GRAY2BGR).astype(np.uint8)
    left_img = cv2.cvtColor(left_mask, cv2.COLOR_GRAY2BGR).astype(np.uint8)
    ori_img = cv2.cvtColor(ori_lbl, cv2.COLOR_GRAY2BGR).astype(np.uint8)
    
    hight = get_hight(right_mask)
    width = get_width(right_mask)
    inertia_image, line_of_inertia_right = inertia_detection_cv_line(right_mask,draw_img=None,nparray=True,scale=int(hight*0.35),width=int(width*0.35),ignore_slope=True)
    inertia_image, line_of_inertia_left = inertia_detection_cv_line(left_mask,draw_img=None,nparray=True,scale=int(hight*0.35),width=int(width*0.35),ignore_slope=True)
    
    line_of_inertia_mid = (np.array(line_of_inertia_right)+np.array(line_of_inertia_left))//2
#     print('line_of_inertia_mid =',line_of_inertia_mid)
    
    merge_angle = get_tooth_degree(np.array(line_of_inertia_mid))
    print('merge_angle =',merge_angle)
    
    cv2.line(right_img, line_of_inertia_right[0],  line_of_inertia_right[1], (255, 0, 255), 2)
    cv2.line(left_img, line_of_inertia_left[0],  line_of_inertia_left[1], (255, 0, 255), 2)
    cv2.line(ori_img, tuple(line_of_inertia_mid[0]),  tuple(line_of_inertia_mid[1]), (255, 0, 255), 2)
    plt.figure(figsize=(10,5))
    plt.subplot(131)
    plt.imshow(ori_img)
    plt.subplot(132)
    plt.imshow(left_img)
    plt.subplot(133)
    plt.imshow(right_img)
    plt.show()
    
    return merge_angle
    
def large_image_corner_detection(components_list):
    new_tmp = []
    tmp = []
    for num in range(len(components_list)):
        aaa = components_list[num].copy().astype('uint8')
        aaa = cv2.resize(aaa,(round(aaa.shape[1]*0.5),round(aaa.shape[0]*0.5)))
        
        if len(np.nonzero(components_list[num])[0]) < 1000:
            continue
        gray = np.float32(aaa)
        dst = cv2.cornerHarris(gray,9,7,0.06) #dst = cv2.cornerHarris(gray,7,7,0.08)
        dst = cv2.dilate(dst,None)

        ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
        dst = np.uint8(dst)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
        for i in range(1, len(corners)):
            if 50 <=int(corners[i,1])<=aaa.shape[0]-50:
                if 30 <=int(corners[i,0])<=aaa.shape[1]-30:
                    cv2.circle(img, (int(corners[i,0])*2, int(corners[i,1])*2), 10, (255,0,0), 3)
                    tmp.append((int(corners[i,0]), int(corners[i,1])))
        for i in range(len(tmp)):
            new_tmp.append((tmp[i][0]*2,tmp[i][1]*2))
    return new_tmp

def turn_red_to_darkpink(result_img):
    for i in range(result_img.shape[0]):
        for j in range(result_img.shape[1]):
            if (result_img[i][j]==(255,0,0)).all():
                result_img[i][j] = (125,0,125)
    return result_img
            
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x), int(y)

def teeth_left_right_most_side(center_point, result_img_ori):
    color_uni = colorful_unique(result_img_ori[center_point[1]-5:center_point[1]+6,center_point[0]-5:center_point[0]+6].astype(int))[0].tolist()
    if len(color_uni) == 2 and ((0,0,0) in color_uni) and ((0,0,255) in color_uni):
        return True
    elif len(color_uni) == 1 and ((0,0,0) in color_uni):
        return True
    elif len(color_uni) == 1 and ((0,0,255) in color_uni):
        return True
    else:
        return False
        
def point_left_right(combined_masks,upper_teeth, result_img):
                
    result_img_ori = result_img.copy()
    remove_list = []
    im_list = []
    select_point = []
    select_point_left = []
    select_point_right = []
    
    
    imgray = combined_masks.copy()

    contours_all = test_function(imgray)
    
    change_times = 0
    change_list = []
    change_point = []
    
#     print(len(contours))
    for contours in contours_all:
        if len(contours)==2:
            continue
        idx_list = []
        
        for idx in range(len(contours)-10):

            imgray = combined_masks.copy()

            im = cv2.cvtColor(imgray,cv2.COLOR_GRAY2BGR)

            point = contours[idx][0]

            cv2.circle(im, tuple(point), 15, (255,0,0), 3)
            cur = point
            center_point = point

            idx_range = []
            for i in range(1,len(contours)-10):
                idx_range.append((idx+i)%len(contours))

            if upper_teeth:
                sign = 1
            else:
                sign = -1

            for cnt in idx_range[::10]:
                if sign*contours[cnt][0][1]<=sign*cur[1]:
                    cur = contours[cnt][0]
                else:
                    break
            cv2.circle(im, tuple(cur), 10, (0,255,255), 3)
            left_point = cur

            cur = point
            for cnt in idx_range[::-1][::10]:
                if sign*contours[cnt][0][1]<=sign*cur[1]:
                    cur = contours[cnt][0]
                else:
                    break
            cv2.circle(im, tuple(cur), 10, (0,255,0), 3)
            right_point = cur

#             plt.imshow(im)
#             plt.show()
            l_distance = abs(euclidean(center_point,left_point))
            r_distance = abs(euclidean(center_point,right_point))
            delete_non_sharp = min(l_distance, r_distance)
            if delete_non_sharp < result_img_ori.shape[1]/50:
                continue
            else:
                if len(idx_list) > 0:
                    if idx - idx_list[-1] < 20:
                        continue

                if center_point[1]-20>=0 and center_point[0]-20 >= 0 and center_point[1]+21 <= result_img_ori.shape[0] and center_point[0]+21 <= result_img_ori.shape[1]:
                    if teeth_left_right_most_side(center_point, result_img_ori):
                        continue
                if center_point[1] <= 20 or center_point[1] >= result_img_ori.shape[0]-20:
                    continue
                    
                if max(abs(center_point[1]-left_point[1]), abs(center_point[1]-right_point[1])) < result_img_ori.shape[0]/20:
#                 if max(l_distance, r_distance) < result_img_ori.shape[0]/10:
                    continue
                    
                max_distance = l_distance + r_distance
                for i in range(-5,6):
                    if i == 0:
                        continue
                    center_p = contours[(idx+i)%len(contours)][0]
                    new_distance = abs(euclidean(center_p, left_point))+abs(euclidean(center_p,right_point))
                    if new_distance > max_distance:
                        center_point = center_p
                select_point.append(center_point)
                select_point_left.append(left_point)
                select_point_right.append(right_point)
                idx_list.append(idx)
                im_list.append(im)
#                 print('select idx =',idx)
        #########################################################
        for center_idx in idx_list:

            imgray = combined_masks.copy()
            im = cv2.cvtColor(imgray,cv2.COLOR_GRAY2BGR)
            

            center_point = contours[center_idx][0]
            cv2.circle(im, tuple(center_point), 15, (255,0,255), 3)
            
            if center_point[1]-20>=0 and center_point[0]-20 >= 0 and center_point[1]+21 <= result_img_ori.shape[0] and center_point[0]+21 <= result_img_ori.shape[1]:
                if teeth_left_right_most_side(center_point, result_img_ori):
                    continue
            else:
                continue
            
            idx_range = []
            for i in range(1,len(contours)-10):
                idx_range.append((center_idx+i)%len(contours))

            last_idx = center_idx
            for cnt in idx_range[::1]:
                center_point = contours[cnt][0]
                
                if center_point[1]-20>=0 and center_point[0]-20 >= 0 and center_point[1]+21 <= result_img_ori.shape[0] and center_point[0]+21 <= result_img_ori.shape[1]:
                    if teeth_left_right_most_side(center_point, result_img_ori):
                        if center_point.tolist() in change_point:
                            break
                            
                        center_point = contours[last_idx][0]
                        cv2.circle(im, tuple(center_point), 15, (255,0,0), 3)
                        change_point.append(center_point.tolist())
                        change_list.append(im)
                        break
                else:
                    break
                last_idx = cnt
            
            last_idx = center_idx
            for cnt in idx_range[::-1][::1]:

                center_point = contours[cnt][0]

                if center_point[1]-20>=0 and center_point[0]-20 >= 0 and center_point[1]+21 <= result_img_ori.shape[0] and center_point[0]+21 <= result_img_ori.shape[1]:
                    if teeth_left_right_most_side(center_point, result_img_ori):
                        if center_point.tolist() in change_point:
                            break
                        center_point = contours[last_idx][0]
                        cv2.circle(im, tuple(center_point), 15, (255,0,0), 3)
                        change_point.append(center_point.tolist())
                        change_list.append(im)
                        break
                else:
                    break
                last_idx = cnt

#         for i in range(len(im_list)):
#             plt.imshow(im_list[i])
#             plt.show()

#         for i in range(len(change_list)):
#             plt.imshow(change_list[i])
#             plt.show()
#     print('change_times =',change_times)
    new_select_point = []
    for point in select_point:
        if point[1]-10>=0 and point[0]-10 >= 0 and point[1]+11 <= result_img_ori.shape[0] and point[0]+11 <= result_img_ori.shape[1]:
            new_select_point.append(point)
    return new_select_point, change_point#,select_point_left,select_point_right


###
# cell
def extend_line_se(s,e):
    scale = 2#0.6 0.8
    vec = e-s
    e = e + scale*vec
    s = s - scale*vec
    return s,e
def extend_line_pol(base_cutline, s,e):
    point1 = (base_cutline[0] + base_cutline[1])//2
    point2 = (base_cutline[2] + base_cutline[3])//2

    dx = point1[0]-point2[0]
    dy = point1[1]-point2[1]

    a = (dx * dx - dy * dy) / (dx * dx + dy*dy)
    b = 2 * dx * dy / (dx*dx + dy*dy)

    x1 = round(a * (s[0] - point1[0]) + b*(s[1] - point1[1]) + point1[0])
    y1 = round(b * (s[0] - point1[0]) - a*(s[1] - point1[1]) + point1[1])
    x2 = round(a * (e[0] - point1[0]) + b*(e[1] - point1[1]) + point1[0])
    y2 = round(b * (e[0] - point1[0]) - a*(e[1] - point1[1]) + point1[1])

#     test = group0_lbl.copy()
#     cv2.line(test, (x1,y1), (x2,y2), (255, 0, 0), 5)
#     cv2.circle(test, (x1,y1), 10, (0,255,0), -1)
#     cv2.circle(test, (x2,y2), 10, (0,255,0), -1)
#     plt.imshow(test)
    return (x1,y1),(x2,y2)

def get_pulp_img_pink(mul_ori):
    image = mul_ori.copy()
    black_pixels_mask = np.all(image == [125, 0, 125], axis=-1)
    non_black_pixels_mask = np.any(image != [125, 0, 125], axis=-1)  
    image[black_pixels_mask] = [255, 255, 255]
    image[non_black_pixels_mask] = [0, 0, 0]
    return image


def test_pulp(mul_ori):
    
    kernel = np.ones((10,10),np.uint8)
    kernel_ero = np.ones((60,15), np.uint8)
    kernel_dila = np.ones((70,15), np.uint8)
    
    result_img = get_pulp_img_pink(mul_ori)
    result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2GRAY)
    result_img = np.where(result_img>128,255,0).astype('uint8')

    group0_cutline = []
    temp_img = result_img.copy()

    result_img = cv2.morphologyEx(result_img, cv2.MORPH_CLOSE, kernel)

    result_img = cv2.dilate(result_img, kernel_dila, iterations = 1)
    result_img = cv2.erode(result_img, kernel_ero, iterations = 1)

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(result_img, connectivity=8)

    sizes = stats[1:, -1]; nb_components = nb_components - 1

    min_size = 200  #1000

    #your answer image
    img2 = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    img2 = img2.astype('uint8')

    img2[0,:] = 0
    img2[img2.shape[0]-1,:] = 0
    img2[:,0] = 0
    img2[:,img2.shape[1]-1] = 0


    img2 = remove_hole(img2)
    #     plt.imshow(img2)
    #     plt.show()

    result_img = img2.copy()
    ################    
    grow = morphology.binary_dilation(result_img, structure=np.ones((100, 40), dtype=int))
    lbl, npatches = measurements.label(grow)
    lbl[result_img==0] = 0


    #=================== the same group============

    unique, counts = np.unique(lbl, return_counts=True)
    key_list = dict(zip(unique, counts))
    components_list = []
    for index, pixel in enumerate(key_list):
        if len(key_list)==1:
            img_gray = lbl.copy()
            component = np.where(img_gray == pixel, 255, 0)
            components_list.append(component)
        elif index != 0 :
            img_gray = lbl.copy()
            component = np.where(img_gray == pixel, 255, 0)
            components_list.append(component)


#     plt.figure(figsize=(10,5))
#     plt.subplot(131)
#     plt.imshow(o_image,cmap='gray')
#     plt.subplot(132)
#     plt.imshow(lbl)
#     plt.subplot(133)
#     plt.imshow(temp_img,cmap='gray')
#     plt.show()


    group0_lbl = lbl.copy()
    group0_lbl[group0_lbl>0] = 255
    group0_lbl = group0_lbl.astype('uint8')
    group0_lbl = cv2.cvtColor(group0_lbl, cv2.COLOR_GRAY2BGR)
    ###############################

    for components_list_number in range(len(components_list)):
        img = components_list[components_list_number].astype(np.uint8)
        img_box = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        long_length_num, group0, boxes,unsort_boxes = find_minrec(img, img_box)
    #         print(long_length_num, group0)

#         if not group0:
#             continue

        img2 = img.copy()
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        starts = []
        ends = []
        for box in boxes:
            start = (box[0]+box[1])//2
            end = (box[2]+box[3])//2
            starts.append(tuple(start))
            ends.append(tuple(end))
            cv2.line(group0_lbl, tuple(start), tuple(end), (0, 255, 0), 4)

        for box in unsort_boxes:            
            cv2.drawContours(group0_lbl, [box], 0, (0,255,0), 2)

        s_x = 0
        s_y = 0
        e_x = 0
        e_y = 0
        for start,end in zip(starts,ends):
            s_x += start[0]
            s_y += start[1]
            e_x += end[0]
            e_y += end[1]


#         if group0:
#             if len(starts)>1:
#                 pass
#             else:
#                 group0_cutline += boxes
        group0_cutline += boxes

    ro_tooth_list = []
    angle_list = []
    if len(group0_cutline)>=1:
        sort_x = [line[:,0].mean() for line in group0_cutline]
        arg_sort_x = np.argsort(sort_x)
        sort_x_cutline = [group0_cutline[x] for x in arg_sort_x] # 綠色
        #########################################################
        for line_num in range(len(sort_x_cutline)):
            angle = get_tooth_degree(np.array(sort_x_cutline[line_num]))
#             print('angle =',angle)
            if angle > 40:
                angle -= 90
            elif angle < -40:
                angle += 90
            angle_list.append(angle)
#         print(angle_list)
        #################################################

        crop_point = []
        flag = False
        for line_num in range(len(sort_x_cutline)-1):
            gap = 2000#o_image.shape[1]*0.5 # 200
            if (abs(sort_x_cutline[line_num][0] + sort_x_cutline[line_num][1]-sort_x_cutline[line_num+1][0] - sort_x_cutline[line_num+1][1])[0]//2 > gap) and (abs(sort_x_cutline[line_num][2] + sort_x_cutline[line_num][3]-sort_x_cutline[line_num+1][2] - sort_x_cutline[line_num+1][3])[0]//2 > gap):
                flag = True
                if len(crop_point)==0:
                    continue
                s = (sort_x_cutline[line_num][0] + sort_x_cutline[line_num][1])#-crop_point[-1][1]
                e = (sort_x_cutline[line_num][2] + sort_x_cutline[line_num][3])#-crop_point[-1][0]
                s,e = extend_line_se(s,e)
                s = s.astype('int') - crop_point[-1][0]
                e = e.astype('int') - crop_point[-1][1]
                cv2.circle(group0_lbl, (s[0],s[1]), 15, (0,255,0), -1)
                cv2.circle(group0_lbl, (e[0],e[1]), 15, (0,255,0), -1)
                crop_point.append((s.tolist(),e.tolist()))
                cv2.line(group0_lbl, (s[0],s[1]), (e[0],e[1]), (255, 0, 0), 4)
                continue


            s = (sort_x_cutline[line_num][0] + sort_x_cutline[line_num][1] + sort_x_cutline[line_num+1][0] + sort_x_cutline[line_num+1][1])//4
            e = (sort_x_cutline[line_num][2] + sort_x_cutline[line_num][3] + sort_x_cutline[line_num+1][2] + sort_x_cutline[line_num+1][3])//4
    #             x1,y1,x2,y2 = extend_line(s,e)

            cv2.circle(group0_lbl, (s[0],s[1]), 15, (0,255,0), -1)
            cv2.circle(group0_lbl, (e[0],e[1]), 15, (0,255,0), -1)

            s,e = extend_line_se(s,e)
            s = s.astype('int')
            e = e.astype('int')
            crop_point.append((s.tolist(),e.tolist()))

            if line_num==0:
                out_point1, out_point2 = extend_line_pol(sort_x_cutline[0], s,e)
                cv2.line(group0_lbl, out_point1, out_point2, (255, 0, 0), 4)
                crop_point.insert(0,(out_point1,out_point2))

            if line_num==len(sort_x_cutline)-2:
                out_point1, out_point2 = extend_line_pol(sort_x_cutline[len(sort_x_cutline)-1], s,e)
                cv2.line(group0_lbl, out_point1, out_point2, (255, 0, 0), 4)
                crop_point.append((out_point1,out_point2))

            if flag == True:
                out_point1, out_point2 = extend_line_pol(sort_x_cutline[line_num], s,e)
                cv2.line(group0_lbl, out_point1, out_point2, (255, 0, 0), 4)
                crop_point.append((out_point1,out_point2)) # NN_160210_103014_C02678 忘記補後面的
                flag = False

            cv2.line(group0_lbl, (s[0],s[1]), (e[0],e[1]), (255, 0, 0), 4)

        sort_line = [line[0][0]+line[1][0] for line in crop_point]
        arg_sort_line = np.argsort(sort_line)
        crop_point = [crop_point[x] for x in arg_sort_line]
    
    return_angle = 0
    
    if angle_list!=[]:
        return_angle = np.array(angle_list).mean()

    o_image_temp = o_image.copy()
    o_image_temp = cv2.cvtColor(o_image_temp, cv2.COLOR_GRAY2BGR)

#     plt.figure(figsize=(16,8))
#     plt.subplot(141)
#     plt.imshow(o_image,cmap='gray')
#     plt.subplot(142)
#     plt.imshow(temp_img,cmap='gray')
#     plt.subplot(143)
#     plt.imshow(group0_lbl)
#     plt.subplot(144)
#     plt.imshow(cv2.addWeighted(o_image_temp,0.5,group0_lbl,0.5,0))
#     plt.show()
    mul_ori_temp = mul_ori.copy()
    for i in range(mul_ori.shape[0]):
        for j in range(mul_ori.shape[1]):
            if (group0_lbl[i][j]==(255,0,0)).all():
                mul_ori_temp[i][j] = (255,0,0)
            elif (group0_lbl[i][j]==(0,255,0)).all():
                mul_ori_temp[i][j] = (48,128,20)
                
#     return cv2.addWeighted(o_image_temp,0.5,group0_lbl,0.5,0)
    return return_angle, mul_ori_temp#cv2.addWeighted(mul_ori,0.5,group0_lbl,0.5,0)


###
# cell
for path in tqdm(img_path_list):
#     path = 'D:/Lab/PBL/tooth_detection/unet/test/seg_image\\' + 'NN_170210_153017_C0A0D2' + '.png'#NN_180719_084849_16F364
#     path = 'D:/Lab/PBL/tooth_detection/unet/test/seg_image/upper\\' + 'NN_191024_151631_BE78BA' + '.png'#NN_180719_084849_16F364
#     path = 'D:/Lab/PBL/tooth_detection/unet/test/seg_image/lower\\' + 'NN_180905_103731_16EC37' + '.png'#NN_180719_084849_16F364

############# preprocessing
    result_img = cv2.imread(path)
    b,g,r = cv2.split(result_img)
    result_img = cv2.merge([r,g,b])
    result_img_gray = cv2.cvtColor(result_img, cv2.COLOR_RGB2GRAY)
    
    result_img = combine_pulp(path,result_img)
    result_img_gray_ori = result_img_gray.copy()
    
    background_component, black_components = find_black_component(result_img_gray)
    
    condition = (result_img_gray==29)# | (result_img_gray==0)
    result_img_gray = np.where(condition,255,0).astype('uint8')
    result_img_gray[black_components==255] = 255
    
    result_img_gray = cv2.dilate(result_img_gray, np.ones((10, 10)), iterations = 1)
    result_img_gray = cv2.erode(result_img_gray, np.ones((10, 10)), iterations = 1)
    result_img_gray = cv2.medianBlur(result_img_gray, 9) #7 9
    
    o_image_path = o_image_folder_path + os.path.basename(path) #至宇的前處理
    o_image_path = glob.glob(o_image_path)
    assert len(o_image_path) > 0
    o_image_folder = ''
    o_image = cv2.imread(o_image_path[0])
    
    result_img_gray = remove_small_component(result_img_gray)    
    result_img_gray[0,:] = 255
    result_img_gray[result_img_gray.shape[0]-1,:] = 255
    result_img_gray[:,0] = 255
    result_img_gray[:,result_img_gray.shape[1]-1] = 255
    
#################### get the mask without background and teeth

    grow = morphology.binary_dilation(result_img_gray, structure=np.ones((1, 1), dtype=int))
    lbl, npatches = measurements.label(grow)
    lbl[result_img_gray==0] = 0
    unique, counts = np.unique(lbl, return_counts=True)
    key_list = dict(zip(unique, counts))
    components_list = []
    for index, pixel in enumerate(key_list):
        if len(key_list)==1:
            img_gray = lbl.copy()
            component = np.where(img_gray == pixel, 255, 0)
#             components_list.append(component)
        elif index != 0 :
            img_gray = lbl.copy()
            component = np.where(img_gray == pixel, 255, 0)
            unique, counts = np.unique(component, return_counts=True)
            if counts[1] < 1000:
                continue
            components_list.append(component)
    img = result_img.copy()
     

    combined_masks = np.zeros((result_img_gray.shape[0],result_img_gray.shape[1]),dtype=np.uint8)
    for i in range(0,len(components_list)):
        if len(np.nonzero(components_list[i])[0]) < 5000:
            continue
        combined_masks[components_list[i]==255]=255
        
#################### decide upper/lower teeth, find the peak points and decide the initial angle

    upper_teeth = False#Is_upper_teeth(combined_masks)

    result_img_ori = result_img.copy()
    result_img = cv2.resize(result_img,(round(result_img.shape[1]*0.25),round(result_img.shape[0]*0.25)), interpolation = cv2.INTER_NEAREST)
     
    o_image = cv2.imread(o_image_path[0],0)
    mul_ori = img.copy()
    result_img_ori_tmp = result_img_ori.copy()
    select_point, change_point = point_left_right(combined_masks,upper_teeth,result_img_ori)
    init_angle, pulp_line = test_pulp(img)
#     print('init_angle =',init_angle)

#################### find out lines and crop teeth

    non_background_component = np.where(background_component>128,0,255)
    non_background_component = cv2.resize(non_background_component,(round(non_background_component.shape[1]*0.25),round(non_background_component.shape[0]*0.25)), interpolation = cv2.INTER_NEAREST).astype('uint8')
   

    point = [(round(point[0]/4),round(point[1]/4)) for point in select_point]
    m_img, line_list = choose_line_draw(point, result_img, non_background_component, init_angle)

    ori_teeth_mask = result_img_ori.copy()
    ori_teeth_mask = cv2.cvtColor(ori_teeth_mask, cv2.COLOR_RGB2GRAY)
    condition = (ori_teeth_mask==29) | (ori_teeth_mask==0) | (ori_teeth_mask==76)
    ori_teeth_mask = np.where(~condition,255,0).astype('uint8')
    
    
    for i in range(len(line_list)):
        cv2.line(mul_ori, line_list[i][0], line_list[i][1], (255,0,0), 4)
    
    crop_teeth_byline(path, o_image, m_img, m_img, upper_teeth, o_image_folder, mul_ori, ori_teeth_mask, result_img_ori)

    
    
##################### plot image

    for p in select_point:
        cv2.circle(mul_ori, tuple(p), 12, (255,0,0), 3)

    
    plt.figure(figsize=(18,9))
    plt.subplot(141)
    plt.imshow(combined_masks,cmap='gray')
    plt.subplot(142)
    plt.imshow(mul_ori)
    plt.subplot(143)
    plt.imshow(pulp_line)
    plt.subplot(144)
    plt.imshow(o_image,cmap = 'gray')
    if not os.path.exists('new_point/'+o_image_folder):
        os.mkdir('new_point/'+o_image_folder)
    plt.savefig('new_point/'+os.path.basename(path)[:-4]+'.png')
#     plt.clf()
#     plt.close('all')
    plt.show()
    
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

