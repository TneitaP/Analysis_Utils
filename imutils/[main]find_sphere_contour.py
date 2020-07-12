from imutils import contours
import imutils
import cv2



if __name__ == "__main__":
    image = cv2.imread("demo_images/20200402-04.tiff") # shapes.png # 20200402-01.tiff
    
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = imutils.auto_canny(gray)
    
    
    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    

    # # loop over the (unsorted) contours and label them
    allowed_mini_area = 10
    alowed_mini_perimeter = 100
    cnt_idx = 0
    for cnt_i in cnts:
        
        M = cv2.moments(cnt_i)
        if  M["m00"] < allowed_mini_area or cnt_i.size < alowed_mini_perimeter: 
            continue

        # deal with this contour:
        print(cv2.isContourConvex(cnt_i))
        orig = contours.label_contour(orig, cnt_i, cnt_idx, color=(240, 0, 159))
        
        x,y,w,h = cv2.boundingRect(cnt_i) 
        orig = cv2.rectangle(orig,(x,y),(x+w,y+h),(255,0,0),2)

        cnt_idx += 1
        

    # # show the original image
    cv2.imshow("Original", orig)
    cv2.waitKey(-1)