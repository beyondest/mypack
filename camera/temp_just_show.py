import control
import cv2



hcamera = control.camera_init()
control.isp_init(hcamera)
camera_info=control.get_all(hcamera)
control.print_getall(camera_info)
#out=control.save_video_camera_init(out_path,name='out.mp4',codec='AVC1')
control.camera_open(hcamera)
pframebuffer_address=control.camera_setframebuffer()
while (cv2.waitKey(1) & 0xFF) != 27:
    
    
    img_ori=control.grab_img(hcamera,pframebuffer_address)
    

    #out.write(dst)
    
    
    cv2.imshow('w1',img_ori) 

cv2.destroyAllWindows()
control.camera_close(hcamera,pframebuffer_address)
#out.release()


