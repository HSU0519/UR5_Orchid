import math

def calculate_angle_soil(left, top, right, bottom, x3, y3, x4, y4):
    # 如果其中一個點為(0, 0)，則不計算角度
    if (x3 == 0 and y3 == 0) or (x4 == 0 and y4 == 0):
        # print("One of the points is (0, 0), angle calculation skipped.")
        angle_in_degrees = 0.0
        
    else:
        # Step 1: Calculate the center x-coordinate of the rectangle
        center_x = (left + right) / 2
    
        # Step 2: Compute the angle of the blue line with respect to the horizontal axis
        delta_x = x4 - x3
        delta_y = y4 - y3
        angle_blue_line = math.atan2(delta_y, delta_x)
    
        # Step 3: Compute the angle between the red vertical line and the blue line
        # Since the red line is vertical, its angle with the horizontal is 90 degrees (π/2 radians)
        angle_vertical_line = math.pi / 2
    
        # Calculate the angle between the vertical red line and the blue line
        angle_between_lines = angle_vertical_line - angle_blue_line
    
        # Convert the angle to degrees
        angle_in_degrees = math.degrees(angle_between_lines)
          
    return angle_in_degrees

def calculate_angle_rotate(left, top, right, bottom, x3, y3, x4, y4):
    # 如果其中一個點為(0, 0)，則不計算角度
    if (x3 == 0 and y3 == 0) or (x4 == 0 and y4 == 0):
        # print("One of the points is (0, 0), angle calculation skipped.")
        angle_in_degrees = 0.0
        
    else:
        # Step 1: Calculate the center x-coordinate of the rectangle
        center_x = (top + bottom) / 2
    
        # Step 2: Compute the angle of the blue line with respect to the horizontal axis
        delta_x = x4 - x3
        delta_y = y4 - y3
        angle_blue_line = math.atan2(delta_y, delta_x)
    
        # Step 3: Compute the angle between the red vertical line and the blue line
        # Since the red line is vertical, its angle with the horizontal is 90 degrees (π/2 radians)
        angle_vertical_line = math.pi
    
        # Calculate the angle between the vertical red line and the blue line
        angle_between_lines = angle_vertical_line - angle_blue_line
    
        # Convert the angle to degrees
        angle_in_degrees = math.degrees(angle_between_lines)
         
    return angle_in_degrees