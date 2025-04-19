# Functions grab_contours() and sort_contours() were not written by me. They are part of the `imutils` library.

import numpy as np
from enum import Enum
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image

#MODEL_PATH    = 'best_model_pool_2_30.keras'
MODEL_PATH    = 'model_improved.keras'
EX_IM_PATH_1  = 'test_img/test_1.jpg'
EX_IM_PATH_2  = 'test_img/test_2.jpg'
EX_IM_PATH_3  = 'test_img/test_3.jpg'
EX_IM_PATH_4  = 'test_img/test_4.jpg'
EX_IM_PATH_5  = 'test_img/test_5.jpg'
EX_IM_PATH_6  = 'test_img/test_6.jpg'
EX_IM_PATH_7  = 'test_img/test_7.jpg'
EX_IM_PATH_8  = 'test_img/test_8.jpg'
EX_IM_PATH_9  = 'test_img/test_9.jpg'
EX_IM_PATH_10 = 'test_img/test_10.jpg'
EX_IM_PATH_11 = 'test_img/test_11.jpg'
EX_IM_PATH_12 = 'test_img/test_12.jpg'
EX_IM_PATH_15 = 'test_img/test_15.jpg'

# Load image
image = cv2.imread(EX_IM_PATH_4)

# Define the dict of characters
character_dict = {0: '一', 1: '七', 2: '三', 3: '上', 4: '下', 5: '不', 6: '东', 7: '么', 8: '九', 9: '习', 10: '书', 11: '买', 12: '了',
                  13: '二', 14: '五', 15: '些', 16: '亮', 17: '人', 18: '什', 19: '今', 20: '他', 21: '们', 22: '会', 23: '住', 24: '作',
                  25: '你', 26: '候', 27: '做', 28: '儿', 29: '先', 30: '八', 31: '六', 32: '关', 33: '兴', 34: '再', 35: '写', 36: '冷',
                  37: '几', 38: '出', 39: '分', 40: '前', 41: '北', 42: '医', 43: '十', 44: '午', 45: '去', 46: '友', 47: '吃', 48: '同',
                  49: '名', 50: '后', 51: '吗', 52: '呢', 53: '和', 54: '哪', 55: '商', 56: '喂', 57: '喜', 58: '喝', 59: '四', 60: '回',
                  61: '国', 62: '在', 63: '坐', 64: '块', 65: '多', 66: '大', 67: '天', 68: '太', 69: '她', 70: '好', 71: '妈', 72: '姐',
                  73: '子', 74: '字', 75: '学', 76: '客', 77: '家', 78: '对', 79: '小', 80: '少', 81: '岁', 82: '工', 83: '师', 84: '年',
                  85: '店', 86: '开', 87: '影', 88: '很', 89: '怎', 90: '想', 91: '我', 92: '打', 93: '日', 94: '时', 95: '明', 96: '星',
                  97: '昨', 98: '是', 99: '月', 100: '有', 101: '朋', 102: '服', 103: '期', 104: '本', 105: '机', 106: '来', 107: '杯',
                  108: '果', 109: '校', 110: '样', 111: '桌', 112: '椅', 113: '欢', 114: '气', 115: '水', 116: '汉', 117: '没', 118: '漂',
                  119: '火', 120: '点', 121: '热', 122: '爱', 123: '爸', 124: '狗', 125: '猫', 126: '现', 127: '生', 128: '电', 129: '的',
                  130: '看', 131: '睡', 132: '租', 133: '站', 134: '米', 135: '系', 136: '老', 137: '能', 138: '脑', 139: '苹', 140: '茶',
                  141: '菜', 142: '衣', 143: '西', 144: '见', 145: '视', 146: '觉', 147: '认', 148: '识', 149: '话', 150: '语', 151: '说',
                  152: '请', 153: '读', 154: '谁', 155: '谢', 156: '起', 157: '车', 158: '这', 159: '那', 160: '都', 161: '里', 162: '钟',
                  163: '钱', 164: '院', 165: '雨', 166: '零', 167: '面', 168: '飞', 169: '饭', 170: '馆', 171: '高'}

class Alignment(Enum):
    """
    Enum to represent the alignment of elements in a layout or structure.

    Attributes:
        UNDER (int): Represents an element that is positioned below another element.
        BESIDE (int): Represents an element that is positioned side-by-side with another element.
        NEITHER (int): Represents an element that is neither positioned below nor adjacent to another element.
    """
    UNDER   = 0
    BESIDE  = 1
    NEITHER = 2 

class CharacterDetector:
    def __init__(self, show_bounding_boxes : bool, h_margin_error : float, w_margin_error : float, gaussian_blur_ksize, gaussian_blur_sigmaX, morph_ksize):
        self.show_bounding_boxes    = show_bounding_boxes
        self.contour_h_margin_error = h_margin_error
        self.contour_w_margin_error = w_margin_error
        self.gaussian_blur_ksize    = gaussian_blur_ksize
        self.gaussian_blur_sigmaX   = gaussian_blur_sigmaX
        self.morph_ksize            = morph_ksize

    def show_prepr_im(self, original_img, blur_im, edge_im, morph_im):
        """
        Displays a 2x2 grid of images to visualize the preprocessing steps applied to an image.

        Parameters:
        - original_img: The original image.
        - blur_im: The blurred version of the image.
        - edge_im: The edge-detected version of the image using Canny.
        - morph_im: Image after using morphological operations to clean up noise and close gaps
        """
        titles = ['Original', 'Blurred', 'Edge-detected', 'After Morphological operations']
        images = [original_img, blur_im, edge_im, morph_im]

        fig, axes = plt.subplots(2, 2, figsize = (8, 5))

        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i], cmap = 'gray')
            ax.set_title(titles[i])
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    def auto_canny(self, image, sigma = 0.33):
        """
        Automatically applies the Canny edge detection algorithm to the input image 
        by dynamically calculating the lower and upper thresholds based on the image's pixel intensity distribution.

        Parameters:
        - image: The input image in grayscale format.
        - sigma: A parameter to control the thresholding sensitivity. Default is 0.33.

        Returns:
        - edged: The output binary image highlighting detected edges.
        """
        # Compute the median of the single channel pixel intensities
        v = np.median(image)

        # Apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))

        edged = cv2.Canny(image, lower, upper)

        return edged

    def sort_contours(self, cnts, method = "left-to-right"):
        """
        Sorts a list of contours based on their position in an image. 
        Sorting can be performed horizontally (left-to-right or right-to-left) or vertically (top-to-bottom or bottom-to-top). 
        
        Parameters:
        - cnts: A list of contours to be sorted. Each contour is a numpy array, typically obtained using cv2.findContours.
        - method: A string indicating the sorting direction. Valid options include:
            - "left-to-right": Sort contours by their x-coordinates in ascending order.
            - "right-to-left": Sort contours by their x-coordinates in descending order.
            - "top-to-bottom": Sort contours by their y-coordinates in ascending order.
            - "bottom-to-top": Sort contours by their y-coordinates in descending order.

        Returns:
        - A tuple containing:
            - cnts: The sorted list of contours.
            - boundingBoxes: The sorted list of bounding boxes corresponding to the contours. Each bounding box is a tuple (x, y, w, h).
        """
        # initialize the reverse flag and sort index
        reverse = False
        i = 0

        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True

        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1

        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]

        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key = lambda b: b[1][i], reverse = reverse))

        # return the list of sorted contours and bounding boxes
        return cnts, boundingBoxes

    def grab_contours(self, contours):
        """
        Processes the output of the OpenCV cv2.findContours function to extract the list of contours, regardless of the OpenCV version. 
        Different versions of OpenCV return varying numbers of values, so this function ensures compatibility by handling cases 
        where the return signature contains 2 or 3 elements.

        Parameters:
        - contours: A tuple returned by cv2.findContours, which can contain:
            - Two elements (contours, hierarchy) in older versions of OpenCV.
            - Three elements (image, contours, hierarchy) in OpenCV v3, v4-pre, or v4-alpha.
        Returns:
        - A list of contours extracted from the tuple, compatible with all OpenCV versions.
        """
        if len(contours) == 2:
            contours = contours[0]

        # if the length of the contours tuple is '3' then we are using either OpenCV v3, v4-pre, or v4-alpha
        elif len(contours) == 3:
            contours = contours[1]

        # otherwise OpenCV has changed their cv2.findContours return signature yet again and I have no idea WTH is going on
        else:
            raise Exception(("Contours tuple must have length 2 or 3, "
                "otherwise OpenCV changed their cv2.findContours return "
                "signature yet again. Refer to OpenCV's documentation "
                "in that case"))

        # return the actual contours array
        return contours

    def preprocess_img(self, original_img, visualize = False):
        """
        Preprocesses the input image by applying a series of transformations to improve edge detection.

        Parameters:
        - original_img: The original image.
        - visualize (optional): A boolean flag that, if set to True, will display the intermediate preprocessing steps (grayscale, blurred, and edge-detected images). Default is False.

        Steps:
        1. Converts the image to grayscale using OpenCV's `cvtColor` function.
        3. Blurs the image using Gaussian blur to reduce noise before edge detection.
        4. Uses Canny edge detection to identify edges in the image.
        5. Applies morphological closing operations to clean up noise and close any gaps in the edges.
        
        Returns:
        - morph: The final preprocessed image after edge detection and morphological operations.
        """
        # Grayscale conversion
        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)  

        # Blurring to smooth the image before edge detection
        blurred = cv2.GaussianBlur(gray, self.gaussian_blur_ksize, self.gaussian_blur_sigmaX)

        # Edge detection with Canny
        edged = self.auto_canny(blurred)

        # Morphological operations to clean up noise and close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.morph_ksize)
        morph  = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        # Show preprocessed images
        if visualize:
            self.show_prepr_im(original_img, blurred, edged, morph)

        return morph

    def save_ROI(image, x, y, w, h, ROI_number):
        """
        Extracts a Region of Interest (ROI) from an input image and saves it as a separate file.

        Parameters:
        - image : The source image from which the ROI will be extracted.
        - x : The x-coordinate of the top-left corner of the ROI.
        - y : The y-coordinate of the top-left corner of the ROI.
        - w : The width of the ROI.
        - h : The height of the ROI.
        - ROI_number : A unique identifier used to name the output file.
        """
        ROI = image[y : y + h, x : x + w]
        cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)

    def get_median_min_max_dimensions(self, contours):
        """
        Calculates the median, minimum, maximum width and height of a list of bounding boxes.

        Parameters:
        - contours: A list of contours detected from the image.
                    Each contour is represented as a list of points (x, y, w, h), where w is width and h is height.

        Returns:
        - median_width: The median of all the widths (w) from the contours.
        - median_height: The median of all the heights (h) from the contours.
        - min_width: The minimum of all the width (w) from the contours.
        - min_height: The minimum of all the heights (h) from the contours.
        - max_width: The maximum of all the width (w) from the contours.
        - max_height: The maximum of all the heights (h) from the contours.
        """
        # Extract widths and heights from the contours
        widths  = [w for _, _, w, _ in contours]
        heights = [h for _, _, _, h in contours]

        # Calculate the median, min, max of widths and heights
        median_width  = np.median(widths)
        median_height = np.median(heights)
        min_width     = min(widths)
        min_height    = min(heights)
        max_width     = max(widths)
        max_height    = max(heights)

        return median_width, median_height, min_width, min_height, max_width, max_height

    def split_right_wrong_contours(self, contours, median_width, median_height, width_margin, height_margin):
        """
        Splits a list of contours into two groups: 'right' contours and 'wrong' contours,
        based on whether their dimensions are within a specified margin of error from the median dimensions.

        Returns:
        - right_contours (list of tuples): A list of contours that meet the width and height criteria.
        - wrong_contours (list of tuples): A list of contours that do not meet the width and height criteria.
        """
        right_contours = []
        wrong_contours = []
        
        for (x, y, w, h) in contours:
            # Check if the contour's width and height are within margin of error from the median values
            if (median_width - width_margin <= w <= median_width + width_margin) and \
            (median_height - height_margin <= h <= median_height + height_margin):
                right_contours.append((x, y, w, h))
            else:
                wrong_contours.append((x, y, w, h))

        return right_contours, wrong_contours

    def do_boxes_overlap(self, box1, box2):
        """
        Checks if two bounding boxes overlap or one is inside the other.
        
        Parameters:
        - box1: A tuple (x, y, w, h) representing the first bounding box.
        - box2: A tuple (x, y, w, h) representing the second bounding box.
        
        Returns:
        - True if the boxes overlap or one is inside the other; otherwise, False.
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate the rectangle boundaries
        end_x1, end_y1 = x1 + w1, y1 + h1
        end_x2, end_y2 = x2 + w2, y2 + h2

        # Check if the boxes overlap
        return not (end_x1 < x2 or end_x2 < x1 or end_y1 < y2 or end_y2 < y1)

    def combine_boxes(self, box1, box2):
        """
        Combines two overlapping or contained bounding boxes into a single bounding box.
        
        Parameters:
        - box1: A tuple (x, y, w, h) representing the first bounding box.
        - box2: A tuple (x, y, w, h) representing the second bounding box.
        
        Returns:
        - A tuple (x, y, w, h) representing the combined bounding box.
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate the combined rectangle boundaries
        start_x = min(x1, x2)
        start_y = min(y1, y2)
        end_x   = max(x1 + w1, x2 + w2)
        end_y   = max(y1 + h1, y2 + h2)

        # Calculate the new width and height
        combined_w = end_x - start_x
        combined_h = end_y - start_y

        return (start_x, start_y, combined_w, combined_h)

    def merge_overlapping_boxes(self, contours):
        """
        Combines overlapping or contained bounding boxes from a list of contours into a new list.
        
        Parameters:
        - contours: A list of bounding boxes, each represented as a tuple (x, y, w, h).
        
        Returns:
        - A new list of bounding boxes after merging overlapping or contained boxes.
        """
        merged_contours = []

        for box in contours:
            if not merged_contours:
                merged_contours.append(box)
                continue

            # Check if the box overlaps with any existing merged contour
            overlap_found = False
            for i, merged_box in enumerate(merged_contours):
                if self.do_boxes_overlap(box, merged_box):
                    # Combine the boxes
                    merged_contours[i] = self.combine_boxes(merged_box, box)
                    overlap_found = True
                    break
            
            # If no overlap was found, add the box as a new contour
            if not overlap_found:
                merged_contours.append(box)

        return merged_contours

    def check_box_alignment(self, box1, box2, gap_threshold = 25, overlap_ratio = 0.5):
        """
        Determines whether two boxes are beside each other or one is under the other.

        Parameters:
        - box1, box2: Tuples representing bounding boxes in the format (x, y, w, h).
        - gap_threshold: Maximum allowed gap (in pixels) to consider boxes "beside" or "under".
        - overlap_ratio: Minimum required overlap ratio (as a fraction) to consider boxes aligned.

        Returns:
        - Alignment.UNDER if the boxes are under each other.
        - Alignment.BESIDE if the boxes are beside each other.
        - Alignment.NEITHER if neither condition is met.
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate the edges of both boxes
        x1_end = x1 + w1
        y1_end = y1 + h1
        x2_end = x2 + w2
        y2_end = y2 + h2

        # Check for "under" alignment
        # Significant horizontal overlap and a small vertical gap
        horizontal_overlap = max(0, min(x1_end, x2_end) - max(x1, x2))
        min_width          = min(w1, w2)

        if horizontal_overlap >= overlap_ratio * min_width:  # At least specified horizontal overlap
            vertical_gap = min(abs(y1_end - y2), abs(y2_end - y1))
            if vertical_gap <= gap_threshold:  # Small vertical gap
                return Alignment.UNDER.value

        # Check for "beside" alignment
        # Significant vertical overlap and a small horizontal gap
        vertical_overlap = max(0, min(y1_end, y2_end) - max(y1, y2))
        min_height       = min(h1, h2)

        if vertical_overlap >= overlap_ratio * min_height:  # At least specified vertical overlap
            horizontal_gap = min(abs(x1_end - x2), abs(x2_end - x1))
            if horizontal_gap <= gap_threshold:  # Small horizontal gap
                return Alignment.BESIDE.value

        return Alignment.NEITHER.value

    def process_wrong_contours(self, wrong_contours_list, median_width, median_height, width_margin, height_margin):
        """
        Processes a list of contours that are incorrectly identified, combining those that can be merged into a single bounding box
        based on alignment and proximity.

        Parameters:
        - wrong_contours_list: List of bounding boxes in the format (x, y, w, h)
        - median_width: The median width of the bounding boxes
        - median_height: The median height of the bounding boxes
        - width_margin: The allowable width margin of error
        - height_margin: The allowable height margin of error

        Steps:
        1. Iterate over all pairs of bounding boxes:
            1. Skip pairs that have already been merged or involve already used indices.
            2. Check their alignment using `check_box_alignment`.
            3. If aligned ("under" or "beside"), calculate the combined dimensions.
            4. Verify if the combined dimensions fall within the acceptable range defined by median dimensions and margins.
            5. If valid, merge the boxes using `combine_boxes` and add the merged box to `combined_contours`.
        2. After processing all pairs, collect unmerged bounding boxes from the original list by excluding used indices.

        Returns:
        - combined_contours: A list of new bounding boxes created by merging pairs.
        - unmerged_contours: A list of bounding boxes that couldn't be merged.
        """
        combined_contours = []
        unmerged_contours = wrong_contours_list.copy()

        used_indices   = set() # To track contours that have been merged
        combined_pairs = [] # To track box pairs that have been merged

        for i, (x_1, y_1, w_1, h_1) in enumerate(wrong_contours_list):

                for j, (x_2, y_2, w_2, h_2) in enumerate(wrong_contours_list):
                    if (x_1 != x_2 or y_1 != y_2) and (i, j) not in combined_pairs and (j, i) not in combined_pairs and i not in used_indices and j not in used_indices:

                        box_alignment = self.check_box_alignment((x_1, y_1, w_1, h_1), (x_2, y_2, w_2, h_2))

                        if box_alignment == 0:
                            if y_1 < y_2:
                                space = y_2 - y_1 - h_1
                            elif y_1 > y_2:
                                space = y_1 - y_2 - h_2
                            else:
                                space = 0
                            
                            combined_h = h_1 + space + h_2
                            combined_w = max(w_1, w_2)
                        
                        elif box_alignment == 1:
                            if x_1 < x_2:
                                space = x_2 - x_1 - w_1
                            elif x_1 > x_2:
                                space = x_1 - x_2 - w_2
                            else:
                                space = 0

                            combined_w = w_1 + space + w_2
                            combined_h = max(h_1, h_2)

                        elif box_alignment == 2:
                            combined_w = None
                            combined_h = None

                        if combined_w != None and combined_h != None:
                            # Check if the combined dimensions are close to the median
                            if (median_width - width_margin <= combined_w <= median_width + width_margin) and \
                                (median_height - height_margin <= combined_h <= median_height + height_margin):

                                start_x, start_y, combined_w, combined_h = self.combine_boxes((x_1, y_1, w_1, h_1), (x_2, y_2, w_2, h_2))
                                
                                combined_contours.append((start_x, start_y, combined_w, combined_h))

                                used_indices.add(i)
                                used_indices.add(j)
                                combined_pairs.append((i, j))
                                combined_pairs.append((j, i))

        unmerged_contours = [wrong_contours_list[k] for k in range(len(wrong_contours_list)) if k not in used_indices]

        return combined_contours, unmerged_contours

    def split_bounding_boxes(self, line, wrong_contours_list, median_height, height_margin, width_to_height_ratio = 2.0, ratio_margin = 0.2):
        """
        Splits bounding boxes in a line into smaller boxes if their width-to-height ratio matches a specified condition, and updates the input lists accordingly.

        Parameters:
        - line: A list of bounding boxes represented as (x, y, w, h).
        - wrong_contours_list: List of bounding boxes in the format (x, y, w, h)
        - median_height: The median height of the bounding boxes.
        - height_margin: The allowable height margin of error.
        - width_to_height_ratio: The expected width-to-height ratio for bounding boxes that should be split. Defaults to 2.0.
        - ratio_margin: The margin of error allowed for the ratio comparison. Defaults to 0.2.

        Returns:
        - line: The updated list of bounding boxes, including any newly created boxes from splitting.
        - wrong_contours_list: The updated list of wrong bounding boxes, with removed entries that have been split.
        """
        for (x, y, w, h) in line[:]:
            ratio = w / h

            if abs(ratio - width_to_height_ratio) <= ratio_margin and (median_height - height_margin <= h <= median_height + height_margin):
                # Split the rectangle into two halves
                half_width = w // 2
                box1 = (x, y, half_width, h)
                box2 = (x + half_width, y, w - half_width, h)

                line.extend([box1, box2])
                line.remove((x, y, w, h))
                wrong_contours_list.remove((x, y, w, h))

        return line, wrong_contours_list

    def group_bounding_boxes_into_lines(self, bounding_boxes, wrong_contours, height_margin, y_threshold = 10):
        """
        Groups a list of bounding boxes into horizontal lines of text based on their vertical proximity,
        and then adds contours from the wrong_contours list into the appropriate lines if their y-coordinate
        falls between the min_y and max_y of the line.

        Parameters:
        - bounding_boxes: List of bounding boxes, where each box is represented as a tuple (x, y, w, h).
        - wrong_contours: List of bounding boxes representing contours to be added, in the same format.
        - height_margin: The allowable height margin of error
        - y_threshold: Maximum vertical distance between bounding boxes to be considered part of the same line. Defaults to 10.

        Returns:
        - lines: A list of lines, where each line is a list of bounding boxes that belong to it, including added wrong_contours.
        """    
        # Sort bounding boxes by their top (y) coordinate
        bounding_boxes = sorted(bounding_boxes, key = lambda box : box[1])

        lines = []
        current_line = [bounding_boxes[0]]

        # Group bounding_boxes into lines
        for i in range(1, len(bounding_boxes)):
            _, y_prev, _, h_prev = current_line[-1]  # Bottom of the last box in the current line
            _, y, _, h = bounding_boxes[i]  # Current box

            # Check if the current box belongs to the current line
            if abs(y - y_prev) <= y_threshold or abs(y + h/2 - (y_prev + h_prev/2)) <= y_threshold:
                current_line.append(bounding_boxes[i])
            else:
                # Finish the current line 
                lines.append(current_line)

                # Start a new line
                current_line = [bounding_boxes[i]]

        # Append the last line
        lines.append(current_line)
        
        # Add wrong_contours to the appropriate lines based on min_y and max_y
        for contour in wrong_contours:
            cx, cy, cw, ch = contour  # Extract coordinates of the contour
            added = False
            for line in lines:
                # Calculate min_y and max_y for the current line
                min_y = min(box[1] for box in line)
                max_y = max(box[1] + box[3] for box in line)

                # Check if the contour's y-coordinate falls within the line's range
                if min_y - height_margin <= cy <= max_y + height_margin:
                    line.append(contour)
                    added = True

            # If no suitable line was found, create a new line for this contour
            if not added:
                lines.append([contour])

        return lines

    def draw_lines_on_image(self, img, lines, color = (255, 0, 0), thickness = 2):
        """
        Draws horizontal lines under groups of bounding boxes on an image.

        Parameters:
        - img: The image on which to draw the lines.
        - lines: A list of lines, where each line is a list of bounding boxes that belong to it.
        - color: The color of the line, in BGR format. Defaults to (255, 0, 0).
        - thickness: The thickness of the line. Defaults to 2.

        Returns:
        - img: The image with the lines drawn on it.
        """
        for line in lines:
            # Calculate the coordinates for the current line
            x_coords = [box[0] for box in line]
            y_coords = [box[1] for box in line]
            widths   = [box[2] for box in line]
            heights  = [box[3] for box in line]

            x_start = min(x_coords)
            y_start = min(y_coords)
            x_end   = max(x + w for x, w in zip(x_coords, widths))
            y_end   = max(y + h for y, h in zip(y_coords, heights))

            # Draw the line
            cv2.line(img, (x_start, y_end), (x_end, y_end), color, thickness)
        
        return img

    def fill_gaps_in_lines(self, lines, wrong_contours_lines, median_width):
        """
        Fills large gaps between bounding boxes in lines by adding new bounding boxes if a contour from the wrong_contours list is detected within the gap.

        Parameters:
        - lines: A list of lines, where each line is a list of bounding boxes that belong to it.
        - wrong_contours_lines: A list of lines, where each line is a list of bounding boxes in the format (x, y, w, h).
        - median_width: The median width of the bounding boxes.

        Returns:
        - updated_lines_with_boxes: The updated list of lines with gaps filled by new bounding boxes.
        """
        updated_lines_with_boxes = []

        for line_idx, line in enumerate(lines):
            line           = sorted(line, key = lambda box : box[0]) # Sort bounding boxes by their x-coordinate
            wrong_contours = wrong_contours_lines[line_idx]  # Get contours for the exact line
            new_line       = [line[0]]

            for i in range(1, len(line)):
                prev_box = new_line[-1]
                curr_box = line[i]

                # Check the horizontal gap between the previous and current bounding box
                prev_x_end   = prev_box[0] + prev_box[2]
                curr_x_start = curr_box[0]
                gap          = curr_x_start - prev_x_end

                # If gap is big, check contours
                if gap > median_width:
                    for wx, wy, ww, wh in wrong_contours:
                        # Check if contour from wrong_contours list is in the gap
                        if prev_x_end <= wx <= curr_x_start:
                            new_box = (prev_x_end + 3, prev_box[1], int(median_width), prev_box[3]) # Create new bounding box
                            new_line.append(new_box)
                            

                new_line.append(curr_box)

            updated_lines_with_boxes.append(new_line)

        return updated_lines_with_boxes

    def add_missing_bounding_boxes(self, lines, wrong_contours, median_width, median_height, width_margin, height_margin):
        """
        Processes lines of bounding boxes to add missing bounding boxes based on wrong contours and fills gaps between bounding boxes where necessary.

        Parameters:
        - lines: A list of lines, where each line is a list of bounding boxes that belong to it.
        - wrong_contours: List of bounding boxes in the format (x, y, w, h).
        - median_width: The median width of the bounding boxes.
        - median_height: The median height of the bounding boxes.
        - width_margin: The allowable width margin of error.
        - height_margin: The allowable height margin of error.

        Returns:
        - result: Updated lines with missing bounding boxes added and gaps filled.
        """
        merged_contours_in_lines       = []
        combined_contours_in_lines     = []
        not_combined_contours_in_lines = []

        for line in lines:
            merged_contours = self.merge_overlapping_boxes(line)

            merged_contours_in_lines.append(merged_contours)

        for line in merged_contours_in_lines:
            combined_contours, not_combined = self.process_wrong_contours(line, median_width, median_height, width_margin, height_margin)

            combined_contours_in_lines.append(combined_contours)
            not_combined_contours_in_lines.append(not_combined)

        all_contours_in_line = [a + b for a, b in zip(combined_contours_in_lines, not_combined_contours_in_lines)]

        with_splitted_contours = []
        wrong_contours_without_splitted = []

        for line in all_contours_in_line:
            line, new_wrong_contours = self.split_bounding_boxes(line, wrong_contours, median_height, height_margin)

            with_splitted_contours.append(line)

            if len(new_wrong_contours) > 0:
                wrong_contours_without_splitted = new_wrong_contours
            else:
                wrong_contours_without_splitted = wrong_contours

        _, _, min_width, min_height, _, _ = self.get_median_min_max_dimensions(wrong_contours_without_splitted)

        # Remove '.', ',', '?', '()' from wrong_contours list
        for (x, y, w, h) in wrong_contours_without_splitted[:]:    
            if (min_width <= w < median_width / 2) and (min_height <= h < median_height):
                wrong_contours_without_splitted.remove((x, y, w, h))

        # Remove common bounding boxes from lines
        wrong_contours_set = set(wrong_contours_without_splitted)
        updated_lines      = []

        # Organize wrong_contours into corresponding lines
        wrong_contours_lines = [[] for _ in range(len(lines))]
        for box in wrong_contours_without_splitted:
            for i, line in enumerate(lines):
                if box in line:
                    wrong_contours_lines[i].append(box)

        for line in with_splitted_contours:
            updated_line = [box for box in line if box not in wrong_contours_set]
            if len(updated_line) > 0:
                updated_lines.append(updated_line)

        result = self.fill_gaps_in_lines(updated_lines, wrong_contours_lines, median_width)
        
        return result

    def detect_contours(self, original_img, morph_img):
        """
        Detects contours in the preprocessed image and filters them based on size, aspect ratio, and area.

        Parameters:
        - original_img: The original image.
        - morph_img: The preprocessed image after edge detection and morphological operations.

        Steps:
        NEED TO REWRITE STEPS
        1. Detects contours in the morphological image using OpenCV's `findContours` function.
        2. Sorts the detected contours from left to right using a custom sorting function (`sort_contours`).
        3. Filters the contours based on width, height, aspect ratio, and area to remove unwanted contours.
        4. Draws green rectangles around the filtered contours on the original image to highlight the detected regions.

        Returns:
        - detected_img: The original image with green rectangles drawn around the detected and filtered contours.
        - right_contours : A list of bounding box dimensions (x, y, w, h) for valid contours.
        """
        detected_image = original_img.copy()
        
        # Detect contours
        cnts = cv2.findContours(morph_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = self.grab_contours(cnts)

        # Sort contours from left to right
        cnts = self.sort_contours(cnts, method = "left-to-right")[0]

        all_contours   = []

        for c in cnts:
            x, y, w, h = cv2.boundingRect(c) # Getting a rectangle around a contour
            all_contours.append((x, y, w, h))

        merged_contours = self.merge_overlapping_boxes(all_contours)

        median_w, median_h, min_w, min_h, max_w, max_h = self.get_median_min_max_dimensions(merged_contours)

        w_margin = median_w * self.contour_w_margin_error
        h_margin = median_h * self.contour_h_margin_error

        right_contours, wrong_contours = self.split_right_wrong_contours(merged_contours, median_w, median_h, w_margin, h_margin)
                
        combined_contours, wrong_contours = self.process_wrong_contours(wrong_contours, median_w, median_h, w_margin, h_margin)

        right_contours += combined_contours

        right_contours = self.merge_overlapping_boxes(right_contours)

        # Remove '.', ',', '?', '()' from wrong_contours list
        for (x, y, w, h) in wrong_contours[:]:    
            if (min_w <= w < median_w / 2) and (min_h <= h < median_h):
                wrong_contours.remove((x, y, w, h))

        # Group bounding boxes into lines
        lines = self.group_bounding_boxes_into_lines(right_contours, wrong_contours, h_margin)

        if len(wrong_contours) > 0:
            result = self.add_missing_bounding_boxes(lines, wrong_contours, median_w, median_h, w_margin, h_margin)
        else:
            result = lines
        
        # Drawing rectangles around each found character
        #ROI_number = 0
        if self.show_bounding_boxes is True:
            for r in result:
                for (x, y, w, h) in r:
                    detected_image = cv2.rectangle(detected_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangles
                    #cv2.putText(detected_image, f'({x}, {y}, {h})', (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

                    #save_ROI(detected_image, x, y, w, h, ROI_number)
                    #ROI_number += 1

            # Draw lines on the image
            detected_image = self.draw_lines_on_image(detected_image, result)
        
        return detected_image, result

class Classifier:
    def __init__(self, model_path, detected_img, resulted_cnts, character_dict, orig_image, show_recodnized_chars : bool):
        self.model          = None
        self.model_path     = model_path
        self.detected_img   = detected_img
        self.resulted_cnts  = resulted_cnts
        self.character_dict = character_dict
        self.orig_image     = orig_image
        self.show_recodnized_chars = show_recodnized_chars

    def recognize_characters(self):
        """
        Recognizes characters in the image by detecting contours, extracting regions of interest (ROIs), and predicting the characters using a trained model.

        Steps:
        1. Detects contours in the preprocessed image and filters them based on size, aspect ratio, and area.
        2. For each detected contour, it extracts the region of interest (ROI) from the original image.
        3. Preprocesses each ROI (grayscale conversion, resizing, and normalization) to match the input format expected by the model.
        4. Uses the model to predict the character within each ROI.
        5. Draws the predicted character on the original image at the location of the contour.

        Returns:
        - annotated_img: The original image with the predicted characters drawn on it, returned in BGR format.
        """
        if self.model is None:
            self.model = load_model(self.model_path, compile = False)

        resulted_chars = []
        
        # Convert the original image to a PIL Image for rendering text
        pil_img = Image.fromarray(cv2.cvtColor(self.detected_img, cv2.COLOR_BGR2RGB))
        draw    = ImageDraw.Draw(pil_img)

        # Load SimSun font
        font = ImageFont.truetype("simsun.ttc", 30)  # Adjust font size as needed
        '''
        ROI_number = 0
        for r in self.resulted_cnts:
            for (x, y, w, h) in r:
                ROI = self.orig_image[y - 5 : y + h + 5, x - 4 : x + w + 1]
                cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
                ROI_number += 1
        '''

        accuracy_list = []
        # Loop through each filtered contour
        for r in self.resulted_cnts:
            resulted_line = []
            for (x, y, w, h) in r:
                # Extract ROI (Region of Interest)
                #roi = self.detected_img[y : y + h, x : x + w]
                roi = self.orig_image[y - 5 : y + h + 5, x - 4 : x + w + 1]

                # Preprocess ROI for the model
                roi_gray       = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                roi_resized    = cv2.resize(roi_gray, (64, 64))  # Resize to (64, 64)
                roi_normalized = roi_resized / 255.0  # Normalize pixel values to [0, 1]
                roi_input      = roi_normalized.reshape(1, 64, 64, 1)  # Add batch and channel dimensions

                # Predict using the model
                prediction      = self.model.predict(roi_input)
                predicted_index = np.argmax(prediction)  # Get the index of the highest probability
                predicted_char  = self.character_dict.get(predicted_index)  # Get character from dictionary

                confidence  = prediction[0][predicted_index]
                print(predicted_char)
                print(confidence)
                print()

                accuracy_list.append(round(confidence, 3))

                # Render the predicted character using PIL and SimSun font
                if self.show_recodnized_chars:
                    draw.text((x, y - 35), predicted_char, font = font, fill = (255, 0, 0, 255))

                resulted_line.append(predicted_char)

            resulted_chars.append(resulted_line)

        # Convert back to OpenCV format
        annotated_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        median_accuracy = np.median(accuracy_list)

        return annotated_img, resulted_chars, median_accuracy
    
    def recognize_single_character(self, image, contour):
        if self.model is None:
            self.model = load_model(self.model_path, compile = False)

        # Use the provided image directly
        # Make sure image is in the right shape (1, 64, 64, 1)
        if len(image.shape) == 4:
            char_img_input = image  # Already in the right shape
        elif len(image.shape) == 3:
            char_img_input = image.reshape(1, image.shape[0], image.shape[1], 1)
        elif len(image.shape) == 2:
            char_img_input = image.reshape(1, image.shape[0], image.shape[1], 1)
        else:
            raise ValueError("Unexpected image shape")

        # Predict
        predictions = self.model.predict(char_img_input)

        # Get the predicted class and confidence
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])

        # Map to character using character_dict
        character = self.character_dict[predicted_class]

        return predicted_class, character, confidence

"""
h_margin_error      = 0.15
w_margin_error      = 0.15
GaussianBlur_ksize  = (1, 3)
GaussianBlur_sigmaX = 0
morph_ksize         = (2, 2)


character_detector = CharacterDetector(True, 0.15, 0.15, (1, 3), 0, (2, 2))

morph = character_detector.preprocess_img(image, True)
detected_img, resulted_contours = character_detector.detect_contours(image, morph)

# Recognize characters with your model and dictionary
classifier = Classifier(MODEL_PATH, detected_img, resulted_contours, character_dict, image, False)
recognized_img, r_chars, median_acc = classifier.recognize_characters()
print(r_chars)

# Show final result
#cv2.imshow("Detected Characters", detected_img)
cv2.imshow("Detected and Recognized Characters", recognized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""