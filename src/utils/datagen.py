import cv2
import numpy as np

from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


def get_random_data (
    annotation_line, 
    input_shape, 
    max_boxes=25, 
    scale=.3, 
    hue=.1, 
    sat=1.5, 
    val=1.5, 
    random=True
    ):

    '''
    random preprocessing for real-time data augmentation
    '''
    
    line = annotation_line.split('\t')
    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    image = cv2.imread(line[0])
    ih, iw, ic = image.shape
    
    if not random:
        resize_scale = min(h/ih, w/iw)

        nw = int(iw * resize_scale)
        nh = int(ih * resize_scale)
        
        max_offx = w - nw
        max_offy = h - nh
        
        dx = max_offx//2
        dy = max_offy//2

        to_x0, to_y0 = max(0, dx),    max(0, dy)
        from_x0, from_y0 = max(0, -dx), max(0, -dy)
        wx, hy = min(w, dx+nw) - to_x0, min(h, dy+nh) - to_y0

        # place image
        image_data = np.zeros((*input_shape,ic), dtype='uint8') + 128
        image_data[to_y0:to_y0+hy, to_x0:to_x0+wx, :] = cv2.resize(image, (nw, nh))[from_y0:from_y0+hy, from_x0:from_x0+wx, :]
        
        flip = False
        image_data = image_data/255.
    else:
        if np.random.uniform() >= 0.5:
            # scale Up
            resize_scale = 1. + scale * np.random.uniform()
            resize_scale = max( h*resize_scale/ih, w*resize_scale/iw)

            nw = int(iw * resize_scale)
            nh = int(ih * resize_scale)

            max_offx = nw - w
            max_offy = nh - h

            dx = int(np.random.uniform() * max_offx)
            dy = int(np.random.uniform() * max_offy)

            # resize and crop
            image = cv2.resize(image, (nw, nh))
            image_data = image[dy : (dy + h), dx : (dx + w), :]

            dx, dy = (-dx, -dy)
        else:
            # scale down
            mul = 1 if np.random.uniform() >= 0.5 else -1

            resize_scale = 1. + mul * scale * np.random.uniform()
            resize_scale = min( h*resize_scale/ih, w*resize_scale/iw)

            nw = int(iw * resize_scale)
            nh = int(ih * resize_scale)

            max_offx = w - nw
            max_offy = h - nh

            dx = int(np.random.uniform() * max_offx)
            dy = int(np.random.uniform() * max_offy)

            to_x0, to_y0 = max(0, dx),    max(0, dy)
            from_x0, from_y0 = max(0, -dx), max(0, -dy)
            wx, hy = min(w, dx+nw) - to_x0, min(h, dy+nh) - to_y0

            # place image
            image_data = np.zeros((*input_shape,ic), dtype='uint8') + 128
            image_data[to_y0:to_y0+hy, to_x0:to_x0+wx, :] = cv2.resize(image, (nw, nh))[from_y0:from_y0+hy, from_x0:from_x0+wx, :]
    
        flip = np.random.uniform() >= 0.5
        if flip: image_data = image_data[:,::-1,:]

        # distort color of the image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = rgb_to_hsv(np.array(image_data)/255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x>1] = 1
        x[x<0] = 0
        image_data = hsv_to_rgb(x) # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes,5))
    if len(box)>0:
        np.random.shuffle(box)
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        if flip: box[:, [0,2]] = w - box[:, [2,0]]
        box[:, 0:2][box[:, 0:2]<0] = 0
        box[:, 2][box[:, 2]>w] = w
        box[:, 3][box[:, 3]>h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
        if len(box)>max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data


def data_generator(annotation_lines, batch_size, input_shape, random):
    '''
    data generator for fit_generator
    '''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for _ in range(batch_size):
            image, box = get_random_data(annotation_lines[i], input_shape, max_boxes=50, random=random)
            image_data.append(image)
            box = box[np.sum(box, axis=1) != 0, :]
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        
        yield image_data, box_data
        
        
def data_generator_wrapper(annotation_lines, batch_size, input_shape, random):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, random)