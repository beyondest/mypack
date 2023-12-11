import serial
import time
import threading

import sys
sys.path.append('../os_op')
from os_op.thread_op import *
from crc import *



    
def read_and_show(ser:serial.Serial,
                  pdata:pos_data):
    read_ori = read_data(ser)
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
                   s_or_a:str = 's'):
    
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
    


    

       
    
    
      

       
        



if __name__ == "__main__":
    
    ser = port_open()
    
    adata = action_data()
    sdata = syn_data()
    pdata = pos_data()
    
    
    task0 = task(0.8,
                 for_circle_func=read_and_show,
                 for_circle_func_deinit=port_close,
                 params_for_circle_func=[ser,pdata],
                 params_for_circle_func_deinit=[ser]
                 )
    
    
    task1 = task(1.5,
                 write_and_show,
                 for_circle_func_deinit=port_close,
                 params_for_circle_func=[ser,adata,sdata,'s'],
                 params_for_circle_func_deinit=[ser]
                 )
    
    
    keyboard_control_task([task0],False)
    
    
    
    
    
    
    
    
    


