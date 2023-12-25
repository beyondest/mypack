import serial
import time
import threading

import sys
sys.path.append('..')
from os_op.thread_op import *
from crc import *
import img.img_operation as imo
import cv2
from motor_params import *
    
def read_and_show(ser:serial.Serial,
                  pdata:pos_data):
    read_ori = read_data(ser)
    if read_ori == b'':
        print("Nothing receive")
    else:
        print(f'receiving: {read_ori}')
        
        if_error =pdata.convert_pos_bytes_to_data(read_ori,if_part_crc=False)
        print(f' if_error:{if_error}, crc_get:{pdata.crc_get},my_crc:{pdata.crc_v}')
        pdata.show()
    
    
def just_read(ser:serial.Serial):
    read_ori = read_data(ser)
    print(f'receiving: {read_ori}')
    
    
    
def write_and_show(ser:serial.Serial,
                   adata:action_data,
                   sdata:syn_data,
                   s_or_a:str ,
                   windows_name:str,
                   debug_trackbar:str,
                   rel_pitch_trackbar:str,
                   rel_yaw_trackbar:str
                   ):
    
    adata.setting_voltage_or_rpm = round(cv2.getTrackbarPos(debug_trackbar,windows_name) - debug_track_bar_scope[1]/2)
    adata.relative_pitch_10000 =round(cv2.getTrackbarPos(rel_pitch_trackbar,windows_name) - pitch_radians_scope[1]/2)
    adata.relative_yaw_10000 = round(cv2.getTrackbarPos(rel_yaw_trackbar,windows_name) - yaw_radians_scope[1]/2)
    
    a_towrite = adata.convert_action_data_to_bytes(if_part_crc=False)
    
    s_towrite = sdata.convert_syn_data_to_bytes(if_part_crc=False)
    if s_or_a == 's':
        
        print(f"Writing: {s_towrite}")
        print(f"crcis:{sdata.crc_v}")
        ser.write(s_towrite)
    elif s_or_a == 'a':
        
        print(f'Writing:{a_towrite}')
        print(f"crcis:{adata.crc_v}")
        
        ser.write(a_towrite)
    
def show_everything(adata:action_data,
                    sdata:syn_data,
                    pdata:pos_data,
                    canvas:imo.Img.canvas,
                    time_show_x_count,
                    debug_value_scope:tuple,
                    windows_name:str,
                    ):
    canvas.img = imo.draw_time_correlated_value(canvas.img,
                                         time_show_x_count,
                                         pdata.present_debug_value,
                                         debug_value_scope,
                                         4)
    txt_x_step = 250
    img = canvas.img.copy()
    txt_x =1
    for i in range(adata.len):
        img = imo.add_text(img,adata.label_list[i],adata.list[i],(txt_x,30),scale_size=0.8)
        txt_x += txt_x_step
        
        
    txt_x =1    
    for i in range(pdata.len):
        img = imo.add_text(img,pdata.label_list[i],pdata.list[i],(txt_x,60),scale_size=0.8)
        txt_x+=txt_x_step

    
    txt_x = 1
    for i in range(sdata.len):
        img = imo.add_text(img,sdata.label_list[i],sdata.list[i],(txt_x,90),scale_size=0.8)
        txt_x+=txt_x_step
      
    
    
    cv2.imshow(windows_name,img)
    
    
def show_deinit():
    cv2.destroyAllWindows()
       
    

      

       
        



if __name__ == "__main__":
    
    ser = port_open()
    
    
    
    
    debug_track_bar = 'tar_rpm'
    rel_pitch_trackbar = 'rel_rad_pit'
    rel_yaw_trackbar = 'rel_rad_yaw'
    windows_name = 'portplot'
    
    
    
    adata = action_data()
    sdata = syn_data()
    pdata = pos_data()
    
    canvas = imo.Img.canvas((700,2000))
    
    time_show_x_count = 0
    
    
    cv2.namedWindow('portplot',cv2.WINDOW_AUTOSIZE)
    
    imo.trackbar_init(debug_track_bar,debug_track_bar_scope,windows_name)
    imo.trackbar_init(rel_pitch_trackbar,pitch_radians_scope,windows_name)
    imo.trackbar_init(rel_yaw_trackbar,yaw_radians_scope,windows_name)
    
    cv2.setTrackbarPos(debug_track_bar,windows_name,int(debug_value_start_pos+debug_track_bar_scope[1]/2))
    cv2.setTrackbarPos(rel_pitch_trackbar,windows_name,int(pitch_start_pos+pitch_radians_scope[1]/2))
    cv2.setTrackbarPos(rel_yaw_trackbar,windows_name,int(yaw_start_pos+yaw_radians_scope[1]/2))


    task1 = task(0.4,
                 read_and_show,
                 port_close,
                 [ser,pdata],
                 [ser])
    
    
    task2 = task(0.7,
                 for_circle_func=write_and_show,
                 params_for_circle_func=[ser,adata,sdata,'a',windows_name,debug_track_bar,rel_pitch_trackbar,rel_yaw_trackbar],
                 )
    


    
    task1.start()
    task2.start()
    while 1:
        time_show_x_count+=1
        if time_show_x_count == canvas.wid:
            time_show_x_count = 0
            canvas.img = imo.Img.canvas((700,2000)).img
        show_everything(adata,sdata,pdata,canvas,time_show_x_count,debug_value_scope,windows_name)
        
        if cv2.waitKey(100) == ord('q'):
            show_deinit()
            task1.end()
            task2.end()
            break
    
    
    
    
    
    
    
    
    


