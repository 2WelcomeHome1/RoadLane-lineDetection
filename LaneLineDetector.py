## Библиотеки
import numpy as np
import cv2 as cv
from numpy.lib.function_base import average, copy
import Lanes

##Запуск видео 

video = cv.VideoCapture ("cv\\Road3.mp4")

## Если видео не открылось
if not video.isOpened (): 
    print ('error')

cv.waitKey (10)


## Покадровый просмотр в цикле
while video.isOpened () :
    _, frame = video.read ()
    
    cv.namedWindow ('Video', cv.WINDOW_NORMAL)
    cv.resizeWindow ('Video', 1200, 1200)
    
    copy_img = np.copy (frame) # Создание копии слоя и наложение
    
    try :
        frame = Lanes.mask (frame)
        frame = Lanes.canny (frame)
        
        lines = cv.HoughLinesP (frame, 2, np.pi/180, 50, np.array ([()]),minLineLength= 50,  maxLineGap= 110) # Доработка линий
        average_lines = Lanes.average_slope_intercept (frame, lines)
        line_image = Lanes.display_lines (copy_img, average_lines)
        combo = cv.addWeighted (copy_img, 0.8, line_image, 0.5, 1)

        cv.imshow ('Video', frame)
    
    except :
        pass

    if cv.waitKey (10) & 0xFF == ord('q') :
        video.release ()
        cv.destroyAllWindows



## Закрытие видеофрагмента
video.release ()
cv.destroyAllWindows