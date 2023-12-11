import serial as ser





usb_path="/dev/ttyUSB0"



def show_only():
    port=ser.Serial(usb_path,baudrate=115100,timeout=1)
    
    try:
        
        while True:
            buffer_size=10
            databuffer=bytearray(buffer_size)
            nums=port.readinto(databuffer)
            print(databuffer)
            
            
    except KeyboardInterrupt:
        print('\n')
        port.close()
        print('port close')
    finally:
        print("exit success")
        
        

if __name__ =="__main__":
    show_only()