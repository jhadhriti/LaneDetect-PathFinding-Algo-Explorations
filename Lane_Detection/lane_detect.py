# Libraries for working with image processing
import numpy as np
import pandas as pd
import cv2

import argparse

def parse_arguments():
    """
    Parse command line arguments.
    Returns:
        args: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Lane detection in images")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--lane_isolate", action="store_true", help="Isolates the lane from the image.")
    parser.add_argument("--gaussian_blur_kernel", type=int, help="Allows you to manually set the kernel size for the Gaussian blur. Use if your output is noisy.")
    return parser.parse_args()


def region_selection(image):
    """
    Determine and cut the region of interest in the input image.
    Parameters:
        image: we pass here the output from canny where we have
        identified edges in the frame
    """
    # create an array of the same size as of the input image
    mask = np.zeros_like(image)
    # if you pass an image with more then one channel
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    # our image only has one channel so it will go under "else"
    else:
          # color of the mask polygon (white)
        ignore_mask_color = 255
    # creating a polygon to focus only on the road in the picture
    # we have created this polygon in accordance to how the camera was placed
    rows, cols = image.shape[:2]
    bottom_left  = [cols * 0.1, rows * 0.95]
    top_left     = [cols * 0.1, rows * 0.70]
    bottom_right = [cols * 1.0, rows * 0.95]
    top_right    = [cols * 1.0, rows * 0.70]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    # filling the polygon with white color and generating the final mask
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # performing Bitwise AND on the input image and mask to get only the edges on the road
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def hough_transform(image):
    """
    Determine and cut the region of interest in the input image.
    Parameter:
        image: grayscale image which should be an output from the edge detector
    """
    # Distance resolution of the accumulator in pixels.
    rho = 1
    # Angle resolution of the accumulator in radians.
    theta = np.pi/180
    # Only lines that are greater than threshold will be returned.
    threshold = 20
    # Line segments shorter than that are rejected.
    minLineLength = 20
    # Maximum allowed gap between points on the same line to link them
    maxLineGap = 500
    # function returns an array containing dimensions of straight lines
    # appearing in the input image
    return cv2.HoughLinesP(image, rho = rho, theta = theta, threshold = threshold,
                           minLineLength = minLineLength, maxLineGap = maxLineGap)

def average_slope_intercept(lines):
    """
    Find the slope and intercept of all lanes in the image.
    Parameters:
        lines: output from Hough Transform
    """
    lane_lines = []  # List to hold all lane lines (slope, intercept)
    weights = []     # List to hold the weights (lengths) of the lines

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            # calculating slope of a line
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 1/2:
              continue
            # calculating intercept of a line
            intercept = y1 - (slope * x1)
            # calculating length of a line
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            lane_lines.append((slope, intercept))
            weights.append(length)

    # Return the average slope and intercept for all lanes
    return lane_lines, weights

def pixel_points(y1, y2, line):
    """
    Converts the slope and intercept of each line into pixel points.
        Parameters:
            y1: y-value of the line's starting point.
            y2: y-value of the line's end point.
            line: The slope and intercept of the line.
    """
    if line is None:
        return None
    slope, intercept = line
    try:
      x1 = int((y1 - intercept)/slope)
      x2 = int((y2 - intercept)/slope)
      y1 = int(y1)
      y2 = int(y2)
      return ((x1, y1), (x2, y2))
    except:
      return((0, int(y1)), (0, int(y2)))

def lane_lines(image, lines):
    """
    Create full length lines from pixel points for all lanes.
    Parameters:
        image: The input test image.
        lines: The output lines from Hough Transform.
    """
    lane_lines, weights = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    full_lines = []
    for lane in lane_lines:
        line = pixel_points(y1, y2, lane)
        if line is not None:
            full_lines.append(line)
    return full_lines


def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
    """
    Draw lines onto the input image.
    Parameters:
        image: The input test image (video frame in our case).
        lines: The output lines from Hough Transform.
        color (Default = red): Line color.
        thickness (Default = 12): Line thickness.
    """
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

def create_lane_bitmap(image, lane_lines):
    """
    Create a bitmap with the regions between multiple lanes marked as black, and the rest as white.
    Parameters:
        image: The original image (for size reference).
        lane_lines: A list of left and right lane segments. Each lane is represented as
                    a tuple ((x1, y1), (x2, y2)).
    Returns:
        A binary (black and white) image with the lane regions marked.
    """
    # Create a blank white image (same size as input)
    height, width = image.shape[:2]
    mask = np.ones((height, width), dtype=np.uint8) * 255  # White image

    if not lane_lines or len(lane_lines) < 2:
        # If not enough lane lines are detected, return the white image
        return mask

    # Extract left and right lane segments
    left_lines = [line for line in lane_lines if line[0][0] < width // 2]  # Left of image center
    right_lines = [line for line in lane_lines if line[0][0] >= width // 2]  # Right of image center

    if not left_lines or not right_lines:
        # If no valid left or right lines are found, return the white image
        return mask

    # Sort the left and right lines by their y-coordinates (from bottom to top)
    left_lines = sorted(left_lines, key=lambda line: line[0][1], reverse=True)
    right_lines = sorted(right_lines, key=lambda line: line[0][1], reverse=True)

    # We will fill the area between corresponding left and right lines
    for left_line, right_line in zip(left_lines, right_lines):
        left_x1, left_y1 = left_line[0]
        left_x2, left_y2 = left_line[1]
        right_x1, right_y1 = right_line[0]
        right_x2, right_y2 = right_line[1]

        # Define the polygon that represents the area between the current left and right lane
        lane_polygon = np.array([
            [left_x1, left_y1],  # Bottom left of the left lane
            [left_x2, left_y2],  # Top left of the left lane
            [right_x2, right_y2],  # Top right of the right lane
            [right_x1, right_y1]  # Bottom right of the right lane
        ], dtype=np.int32)

        # Fill the region between the lanes with black (0 value)
        cv2.fillPoly(mask, [lane_polygon], 0)  # Black polygon in the mask

    return mask

def process_frame_for_bitmap(image,gaussian_blur=3):
    """
    Process the frame to detect lanes and create a bitmap showing lane regions.
    """
    kernel_size = gaussian_blur
    blur = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    # Detect lane lines in the frame (you would have this from your lane detection code)
    edges = cv2.Canny(blur, 50, 150)  # For illustration, applying edge detection
    region = region_selection(edges)  # Focus on the road region
    lines = hough_transform(region)  # Get lines using Hough Transform
    lane_line_segments = lane_lines(image, lines)  # Get all detected lane line segments

    # Create bitmap showing lane regions
    bitmap_output = create_lane_bitmap(image, lane_line_segments)

    return bitmap_output

def restore_lane_region(image, lane_mask):
    """
    Retain the lane regions from the original image and keep the non-lane regions as white.

    Parameters:
        image: The original image.
        lane_mask: The binary lane mask where lane regions are black (0), and the rest is white (255).

    Returns:
        An image with the original lane regions and white background elsewhere.
    """
    # Create a copy of the original image
    restored_image = np.ones_like(image) * 255  # White background

    # Invert the lane mask (so that the lane regions are 255, and the rest is 0)
    inverted_mask = cv2.bitwise_not(lane_mask)

    # Use the inverted mask to retain the lane regions in the original image
    lane_regions = cv2.bitwise_and(image, image, mask=inverted_mask)

    # Combine the white background and the lane regions
    restored_image = np.where(lane_regions != 0, lane_regions, restored_image)

    return restored_image

if __name__ == "__main__":
    args = parse_arguments()
    image = cv2.imread(args.image_path)
    if args.gaussian_blur_kernel:
        lane_mask = process_frame_for_bitmap(image, args.gaussian_blur_kernel)
    else:
        lane_mask = process_frame_for_bitmap(image)
    if args.lane_isolate:
        isolated_lane_image = restore_lane_region(image, lane_mask)
        cv2.imwrite('isolated_lane_image.png', isolated_lane_image)
        print("Isolated lane image saved as isolated_lane_image.png")
    else:
        cv2.imwrite('lane_mask.png', lane_mask)
        print("Lane mask image saved as lane_mask.png")