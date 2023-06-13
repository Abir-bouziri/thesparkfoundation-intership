import cv2
import pandas as pd
import numpy as np
cap = cv2.VideoCapture(0)

while(True):
  ret, img1 = cap.read()
 


#green 
# to determin the boundary of color BgR to hsv
  color = np.uint8([[[0,255,0]]])

# convert the color to HSV
  hsvcolor = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
  print(hsvcolor)

# convert to hsv colorspace
  hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

# lower bound and upper bound for Green color
  lower_bound = np.array([hsvcolor[0][0][0] - 20, 100, 100])	 
  upper_bound = np.array([hsvcolor[0][0][0] + 20, 255, 255])

  print(lower_bound)
  print(upper_bound)

# find the colors within the boundaries
  mask = cv2.inRange(hsv, lower_bound, upper_bound)

#define kernel size  
  kernel = np.ones((9,9),np.uint8)

# Remove unnecessary noise from mask

  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


# Segment only the detected region
  segmented_img = cv2.bitwise_and(img1,img1, mask=mask)



# Find contours from the mask
  contours,h= cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# output = cv2.drawContours(img1,contours,-1,(0,255,0),3)   
  for pic, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if(area > 300):
       x,y, w, h = cv2.boundingRect(contour)
       img1 = cv2.rectangle(img1, (x, y), 
                                       (x + w, y + h), 
                                       (0, 255,0), 3)
    #img1=cv2.drawContours(img1,contours,-1,(0,0,255),3)   
       cv2.putText(img1, "Green",(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,255,0), 2)   
    
    
# blue
# to determin the boundary of color BgR to hsv
  color2 = np.uint8([[[255,0,0]]])

# convert the color to HSV
  hsvcolor2 = cv2.cvtColor(color2, cv2.COLOR_BGR2HSV)
  print(hsvcolor2)

# lower bound and upper bound for Green color
  lower_bound = np.array([hsvcolor2[0][0][0] -20, 100, 100])	 
  upper_bound = np.array([hsvcolor2[0][0][0] + 20, 255, 255])

  print(lower_bound)
  print(upper_bound)

# find the colors within the boundaries
  mask2 = cv2.inRange(hsv, lower_bound, upper_bound)

# Remove unnecessary noise from mask
  mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
  mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)



# Segment only the detected region
  segmented_img = cv2.bitwise_and(img1,img1, mask=mask2)


# Find contours from the mask
  contours1,h1= cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# output = cv2.drawContours(img1,contours,-1,(0,255,0),3)   
  for pic, contour in enumerate(contours1):
    area = cv2.contourArea(contour)
    if(area > 300):
       x1, y1, w1, h1 = cv2.boundingRect(contour)
       img1 = cv2.rectangle(img1, (x1, y1), 
                                       (x1 + w1, y1 + h1), 
                                       (255,0,0), 3)  
       cv2.putText(img1, "Blue",(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(255,0,0), 2)   
  

    
# red

# lower bound and upper bound for Green color
  lower_bound = np.array([170, 100, 100])	 
  upper_bound = np.array([190, 255, 255])

  print(lower_bound)
  print(upper_bound)

# find the colors within the boundaries
  mask3 = cv2.inRange(hsv, lower_bound, upper_bound)


# Remove unnecessary noise from mask

  mask3 = cv2.morphologyEx(mask3, cv2.MORPH_CLOSE, kernel)
  mask3 = cv2.morphologyEx(mask3, cv2.MORPH_OPEN, kernel)


# Segment only the detected region
  segmented_img = cv2.bitwise_and(img1,img1, mask=mask3)


# Find contours from the mask
  contours2,h2= cv2.findContours(mask3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# output = cv2.drawContours(img1,contours,-1,(0,255,0),3)   
  for pic, contour in enumerate(contours2):
    area = cv2.contourArea(contour)
    if(area > 300):
       x2, y2, w2, h2 = cv2.boundingRect(contour)
       img1 = cv2.rectangle(img1, (x2, y2), 
                                       (x2 + w2, y2 + h2), 
                                       (0,0,255), 2)
    #img1=cv2.drawContours(img1,contours,-1,(255,230,255),3)      
       cv2.putText(img1, "Red",(x2,y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,0,255), 2)        
  
#yellow


# lower bound and upper bound for Green color
  lower_bound = np.array([20 , 100, 100])	 
  upper_bound = np.array([30, 255, 255])

  print(lower_bound)
  print(upper_bound)
# find the colors within the boundaries
  mask4 = cv2.inRange(hsv, lower_bound, upper_bound)


# Remove unnecessary noise from mask

  mask4 = cv2.morphologyEx(mask4, cv2.MORPH_CLOSE, kernel)
  mask4 = cv2.morphologyEx(mask4, cv2.MORPH_OPEN, kernel)


# Segment only the detected region
  segmented_img = cv2.bitwise_and(img1,img1, mask=mask4)

# Find contours from the mask
  contours3,h3= cv2.findContours(mask4.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# output = cv2.drawContours(img1,contours,-1,(0,255,0),3)   
  for pic, contour in enumerate(contours3):
      area = cv2.contourArea(contour)
      if(area > 300):
        x3, y3, w3, h3 = cv2.boundingRect(contour)
        img1 = cv2.rectangle(img1, (x3, y3), 
                                       (x3 + w3, y3 + h3), 
                                       (0,255,255), 3)
    #img1=cv2.drawContours(img1,contours,-1,(255,230,255),3)   
        cv2.putText(img1, "Yellow",(x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,255,255), 2)

#orange
# lower bound and upper bound for Green color
  lower_bound = np.array([5, 100, 100])	 
  upper_bound = np.array([10, 255, 255])

  print(lower_bound)
  print(upper_bound)
# find the colors within the boundaries
  mask5= cv2.inRange(hsv, lower_bound, upper_bound)


# Remove unnecessary noise from mask

  mask5= cv2.morphologyEx(mask5,cv2.MORPH_CLOSE, kernel)
  mask5= cv2.morphologyEx(mask5,cv2.MORPH_OPEN, kernel)


# Segment only the detected region
  segmented_img = cv2.bitwise_and(img1,img1, mask=mask5)

# Find contours from the mask
  contourso,ho= cv2.findContours(mask5.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# output = cv2.drawContours(img1,contours,-1,(0,255,0),3)   
  for pic, contour in enumerate(contourso):
      area = cv2.contourArea(contour)
      if(area > 300):
        xo, yo, wo, ho = cv2.boundingRect(contour)
        img1 = cv2.rectangle(img1, (xo, yo), 
                                       (xo + wo, yo + ho), 
                                       (0,165,255), 3)
    #img1=cv2.drawContours(img1,contours,-1,(255,230,255),3)   
        cv2.putText(img1, "Orange",(xo, yo - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,165,255), 2) 



#Rose
# lower bound and upper bound for Green color
  lower_bound = np.array([150, 70, 50])	 
  upper_bound = np.array([170, 255, 255])

  print(lower_bound)
  print(upper_bound)
# find the colors within the boundaries
  mask6= cv2.inRange(hsv, lower_bound, upper_bound)


# Remove unnecessary noise from mask

  mask6= cv2.morphologyEx(mask6,cv2.MORPH_CLOSE, kernel)
  mask6= cv2.morphologyEx(mask6,cv2.MORPH_OPEN, kernel)


# Segment only the detected region
  segmented_img = cv2.bitwise_and(img1,img1, mask=mask6)

# Find contours from the mask
  contoursp,hp= cv2.findContours(mask6.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# output = cv2.drawContours(img1,contours,-1,(0,255,0),3)   
  for pic, contour in enumerate(contoursp):
      area = cv2.contourArea(contour)
      if(area > 300):
        xp, yp, wp, hp = cv2.boundingRect(contour)
        img1 = cv2.rectangle(img1, (xp, yp), 
                                       (xp + wp, yp + hp), 
                                       (203,192,255), 3)
    #img1=cv2.drawContours(img1,contours,-1,(255,230,255),3) 
        cv2.putText(img1, "PINK",(xp, yp - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(203,192,255), 2)    

# white
# lower bound and upper bound for Green color
  lower_bound = np.array([0, 0, 200])	 
  upper_bound = np.array([180, 30, 255])

  print(lower_bound)
  print(upper_bound)
# find the colors within the boundaries
  mask7= cv2.inRange(hsv, lower_bound, upper_bound)


# Remove unnecessary noise from mask

  mask7= cv2.morphologyEx(mask7,cv2.MORPH_CLOSE, kernel)
  mask7= cv2.morphologyEx(mask7,cv2.MORPH_OPEN, kernel)


# Segment only the detected region
  segmented_img = cv2.bitwise_and(img1,img1, mask=mask7)

# Find contours from the mask
  contoursw,hw= cv2.findContours(mask7.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# output = cv2.drawContours(img1,contours,-1,(0,255,0),3)   
  for pic, contour in enumerate(contoursw):
      area = cv2.contourArea(contour)
      if(area > 300):
        xw, yw, ww, hw = cv2.boundingRect(contour)
        img1 = cv2.rectangle(img1, (xw, yw), 
                                       (xw + ww, yw + hw), 
                                       (0,0,0), 3)
        cv2.putText(img1, "WHITE",(xw, yw - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0, 0), 2)

# Black 
# lower bound and upper bound for Green color
  lower_bound = np.array([0, 0, 0])	 
  upper_bound = np.array([180, 255, 30])

  print(lower_bound)
  print(upper_bound)
# find the colors within the boundaries
  mask7= cv2.inRange(hsv, lower_bound, upper_bound)


# Remove unnecessary noise from mask

  mask7= cv2.morphologyEx(mask7,cv2.MORPH_CLOSE, kernel)
  mask7= cv2.morphologyEx(mask7,cv2.MORPH_OPEN, kernel)


# Segment only the detected region
  segmented_img = cv2.bitwise_and(img1,img1, mask=mask7)

# Find contours from the mask
  contoursw,hw= cv2.findContours(mask7.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# output = cv2.drawContours(img1,contours,-1,(0,255,0),3)   
  for pic, contour in enumerate(contoursw):
      area = cv2.contourArea(contour)
      if(area > 300):
        xw, yw, ww, hw = cv2.boundingRect(contour)
        img1 = cv2.rectangle(img1, (xw, yw), 
                                       (xw + ww, yw + hw), 
                                       (255,255,255), 3)
        cv2.putText(img1, "Black",(xw, yw - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)







  cv2.imshow('frame', img1)
  if cv2.waitKey(1) & 0xFF == ord('q'):
   break
 
cap.release()
cv2.destroyAllWindows()