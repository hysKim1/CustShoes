
import cv2
import numpy as np
import sys
import io


path = "img/aa.jpeg"
BLUE, GREEN, RED, BLACK, WHITE = (255,0,0), (0,255,0),(0,0,255),(0,0,0),(255,255,255)
DRAW_BG = {'color':BLACK, 'val':0}
DRAW_FG = {'color':WHITE, 'val':1}

rect = (0,0,1,1)
drawing = False
rectangle = False
rect_over = False
rect_or_mask = 100
value = DRAW_FG
thickness = 3


def onMouse(event, x, y, flags, param):
    global ix, iy, img, img2, drawing, value, mask, rectangle
    global rect, rect_or_mask, rect_over

    if event == cv2.EVENT_RBUTTONDOWN:
        rectangle = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:

        if rectangle:
            img = img2.copy()
            cv2.rectangle(img,(ix,iy),(x,y),RED,2)
            rect = (min(ix,x), min(iy,y), abs(ix-x), abs(iy-y))
            rect_or_mask = 0
 
    elif event == cv2.EVENT_RBUTTONUP:
        rectangle = False
        rect_over = True
        cv2.rectangle(img, (ix,iy),(x,y),RED,2)
        rect = (min(ix,x),min(iy,y), abs(ix-x), abs(iy-y))
        rect_or_mask = 0
        print("n: 적용 ")

    if event == cv2.EVENT_LBUTTONDOWN:
        if not rect_over:
            print("전경 선택")

        else:
            drawing = True
            cv2.circle(img,(x,y),thickness, value['color'], -1)

            cv2.circle(mask,(x,y),thickness, value['val'], -1)


    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing:
            cv2.circle(img,(x,y),thickness,value['color'], -1)
 
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv2.EVENT_LBUTTONUP:
        
        if drawing:
            drawing = False
            
            cv2.circle(img,(x,y),thickness,value['color'], -1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

    return

def grabcut(path):
    '''
    사용자 
    n: grabcut 결과물 반영
    0: 제거할 배경 선택(검정) background to remove
    1: 유지할 전경 선택(흰색)forground to remain
    r:reset
    esc: 끝내기
    '''
    global ix, iy, img, img2, drawing, value, mask, rectangle
    global rect, rect_or_mask, rect_over   
    img = cv2.imread(path)
    img2 = img.copy()

   
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    output = np.zeros(img.shape, np.uint8)


    cv2.namedWindow('input')
    cv2.namedWindow('output')
    cv2.setMouseCallback('input',onMouse,param=(img,img2))

    print("오른쪽 마우스 버튼을 누르고 영역을 지정한 후 n 을 누르시오.")

    while True:
        cv2.imshow('output',output)
        cv2.imshow('input',img)

        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            break

        elif k == ord('0'):
            print('왼쪽 마우스로 제거할 부분을 표시한 후 n을 누르세요')
            value = DRAW_BG

 
        elif k == ord('1'):
            print('왼쪽 마우스로 복원할 부분을 표시한 후 n을 누르세요')
            value = DRAW_FG

  
        elif k == ord('r'):
            
            print("리셋합니다")
            rect = (0,0,1,1)
            drawing = False
            rectangle = False
            rect_or_mask = 100
            rect_over = False
            value = DRAW_FG
            img = img2.copy()
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            output = np.zeros(img.shape, np.uint8)
            print('0 : 제거배경선택   1: 복원전경선택   n:적용하기   r:리셋하기')

      
        elif k == ord('n'):
            bgdModel = np.zeros((1,65),np.float64)
            fgdModel = np.zeros((1,65),np.float64)

            if rect_or_mask == 0:
                cv2.grabCut(img2,mask,rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
                rect_or_mask = 1

            elif rect_or_mask == 1:
                cv2.grabCut(img2,mask,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_MASK)

            print('0 : 제거배경선택   1: 복원전경선택   n:적용하기   r:리셋하기')

        mask2 = np.where((mask==1) + (mask==3), 255, 0).astype('uint8')
        output = cv2.bitwise_and(img, img2, mask=mask2)
