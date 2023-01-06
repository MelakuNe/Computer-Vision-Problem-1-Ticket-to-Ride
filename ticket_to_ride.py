import numpy as np
import cv2 as cv

def predict_image(img: np.ndarray):
    
#     centers = np.array([x_coor, y_coord])   # city coordinate pairs
    centers, _ = template_matching(img)    
    
    # number of trains corresponding to each color in the image
    # scores of each color for the whole image, just get the score for the separate route then get summ for each color
    n_trains, scores = count_train_AND_scores(img)

    
    return centers, n_trains, scores

def template_matching(image, image2=None):  
    '''
    choose template from the image 'train/all.jpg' which will be used for all other images 
    as a template
    '''
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    im = cv.imread('train/all.jpg', 0)
    template_find = im[2300:2500, 300:500]
    template_final = template_find[40:95, 93:131]
    
    height, width = template_final.shape[0], template_final.shape[1] 
    output = cv.matchTemplate(image, template_final, cv.TM_CCOEFF_NORMED)
    cut_point = 0.63
    find_one_in = 7
    cities = np.where( output >= cut_point)
    all_center = [[0, 0]]
    i = 0
    for coord in zip(*cities):
        center = (coord[0] + int(width/2), coord[1] + int(height/2))
        radius = int(height/2)
        if image2 is not None:
            cv.circle(image2, center, radius, (0,255,0))   
        if (center[0] - all_center[i][0]) > find_one_in  or (center[1] - all_center[i][1]) > find_one_in:
            all_center.append(list(center))
            i += 1
    all_center = all_center[1:]
    return all_center, image2


# separate object from background
# separate object from background
def mask_colors(image, color):
    HLS = cv.cvtColor(image, cv.COLOR_BGR2HLS)
    HUE = HLS[:, :, 0]           
    LIGHT = HLS[:, :, 1]
    SAT = HLS[:, :, 2]
    if color == 'blue':
        mask = (HUE < 105) & (HUE > 90) & (SAT > 140) & (SAT < 250) & (LIGHT > 0) & (LIGHT < 70)
    elif color == 'green':
        mask = (HUE < 90) & (HUE > 60) 
    elif color == 'black':
            mask = (LIGHT < 25) & (SAT < 50)
    elif color == 'yellow':
            mask = (HUE > 22) & (HUE < 28) & (SAT > 139.8) & (SAT < 180)
    elif color == 'red':
            mask = (HUE < 180) & (HUE > 175) & (SAT > 140) & (SAT < 200)
    
    return mask

def morphology(mask, color):
    All_Kernel = {"blue" : (18, 22), "green" : (10, 15), "black" : (10, 25), "yellow" : (4, 15), "red" : (10, 18)}
    mask = mask.astype(np.uint8)
    x, y = All_Kernel[color][0], All_Kernel[color][1]
    ker_close = np.ones((x, x),np.uint8)
    
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, ker_close)
    
    ker_open = np.ones((y, y), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, ker_open)
    
    return mask

def count_train_AND_scores(image):
    score_check = {1: 1, 2: 2, 3: 4, 4: 7, 6: 15, 8: 21}
    COLORS = ('blue', 'green', 'black', 'yellow', 'red')
    n_trains = {'blue': [], 'green': [], 'black': [], 'yellow': [], "red": []}
    scores = {'blue': [], 'green': [], 'black': [], 'yellow': [], 'red': []}
    for color in COLORS:
        mask = mask_colors(image, color)
        pure_mask = morphology(mask, color)
        contours, hierarchy = cv.findContours(pure_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        trains = 0
        all_perimeters = []
        for contour in contours:
            per = cv.arcLength(contour, True)
            all_perimeters.append(per)
            trains += per
       
        get_trains = int(round((trains / 315), 0))
        n_trains[color].append(get_trains)
        n_trains[color] = int(n_trains[color][0])
        
        if color == 'blue':
            all_perimeters = [i/185 for i in all_perimeters]
        elif color == 'green':
            all_perimeters = [i/220 for i in all_perimeters]
        elif color == 'black':
            all_perimeters = [i/215 for i in all_perimeters]
        elif color == 'yellow':
            all_perimeters = [i/200 for i in all_perimeters]
        elif color == 'red':
            all_perimeters = [i/232 for i in all_perimeters]
            
        sco = np.sum(all_perimeters)
        sco = int(round(sco,0))
        scores[color].append(sco)
        scores[color] = int(scores[color][0])
    

    return n_trains, scores