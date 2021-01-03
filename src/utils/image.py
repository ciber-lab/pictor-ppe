import cv2
import numpy as np
import matplotlib as mpl


def letterbox_image(image, size):
    '''
    Resize image with unchanged aspect ratio using padding
    '''

    # original image size
    ih, iw, ic = image.shape

    # given size
    h, w = size

    # scale and new size of the image
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    
    # placeholder letter box
    new_image = np.zeros((h, w, ic), dtype='uint8') + 128

    # top-left corner
    top, left = (h - nh)//2, (w - nw)//2

    # paste the scaled image in the placeholder anchoring at the top-left corner
    new_image[top:top+nh, left:left+nw, :] = cv2.resize(image, (nw, nh))
    
    return new_image


def draw_detection(
    img,
    boxes,
    class_names,
    # drawing configs
    font=cv2.FONT_HERSHEY_DUPLEX,
    font_scale=0.5,
    box_thickness=2,
    border=5,
    text_color=(255, 255, 255),
    text_weight=1
):
    '''
    Draw the bounding boxes on the image
    '''
    # generate some colors for different classes
    num_classes = len(class_names) # number of classes
    colors = [mpl.colors.hsv_to_rgb((i/num_classes, 1, 1)) * 255 for i in range(num_classes)]
    
    # draw the detections
    for box in boxes:
        x1, y1, x2, y2 = box[:4].astype(int)
        score = box[-2]
        label = int(box[-1])

        clr = colors[label]

        # draw the bounding box
        img = cv2.rectangle(img, (x1, y1), (x2, y2), clr, box_thickness)

        # text: <object class> (<confidence score in percent>%)
        text = f'{class_names[label]} ({score*100:.0f}%)'

        # get width (tw) and height (th) of the text
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)

        # background rectangle for the text
        tb_x1 = x1 - box_thickness//2
        tb_y1 = y1 - box_thickness//2 - th - 2*border
        tb_x2 = x1 + tw + 2*border
        tb_y2 = y1

        # draw the background rectangle
        img = cv2.rectangle(img, (tb_x1, tb_y1), (tb_x2, tb_y2), clr, -1)

        # put the text
        img = cv2.putText(img, text, (x1 + border, y1 - border), font, font_scale, text_color, text_weight, cv2.LINE_AA)

    return img