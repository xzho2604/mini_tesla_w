from collections import OrderedDict
import numpy as np
import socket
import RPi.GPIO as GPIO
import time
# Import the PCA9685 module.
import Adafruit_PCA9685
import picamera

#pi camera config
camera = picamera.PiCamera()
camera.resolution = (240, 120)
camera.color_effects = (128,128) #change effect to black and white
camera.framerate=10


# Initialise the PCA9685 using the default address (0x40).
pwm = Adafruit_PCA9685.PCA9685()

servo_stop = 360   # Min pulse length out of 4096
servo_forward = 390 # Max pulse length out of 4096
servo_back = 335

servo_mid = 400
servo_left = 460
servo_right = 350


#======================================================================================
# Helper function to make setting a servo pulse width simpler.
def set_servo_pulse(channel, pulse):
    pulse_length = 1000000    # 1,000,000 us per second
    pulse_length //= 60       # 60 Hz
    print('{0}us per period'.format(pulse_length))
    pulse_length //= 4096     # 12 bits of resolution
    print('{0}us per bit'.format(pulse_length))
    pulse *= 1000
    pulse //= pulse_length
    pwm.set_pwm(channel, 0, pulse)

# Set frequency to 60hz, good for servos.
pwm.set_pwm_freq(60)
print('Start listening to commnand')

#function to process data control car and record image with label
def car_run(data):
    pass


#======================================================================================
def main():
    #create tcp socket
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.bind(("",5555))
    tcp_socket.listen()

    client_socket,client_addr = tcp_socket.accept()

    #init the file name var
    index = 0  #init the file name index
    path = '/home/pi/Desktop/data/' #the folder to save the labeled image

    #init the control dict and vector
    control_dict = {'i':0,'k':1,'l':2,'j':3,' ':4}
    prev_control_vector = np.array([0,0,0,0,0])
    #control index:[channel,servo_val] key at index pressed
    control_action_on = {0:[0,servo_forward], 1:[0,servo_back], 2:[1,servo_right], 3:[1,servo_left]}
    control_action_off = {0:[0,servo_stop], 1:[0,servo_stop], 2:[1,servo_mid], 3:[1,servo_mid]}

    while True:
        data = client_socket.recv(4096)
        data = data.decode('UTF-8')
        if data: #when there is command
            #print(data) #if there is command print command
            rev_list = [x for x in list(data) if x != 'n'] #remove 'n'
            rev_list =np.array(list(OrderedDict.fromkeys(rev_list))) #remove duplicates
            #print(rev_list)

            #convert the received current key stauts into control vectors
            #find control index of the received control status
            control_index = np.array([int(control_dict[x]) for x in rev_list])
            curr_control_vector = np.array([0,0,0,0,0]) #init

            #if there is any control since n is place holder
            if control_index.size != 0:
                curr_control_vector[control_index] = 1 #assign active control to 1
                print(curr_control_vector)

                #now we have the curr_control vector compare with the previous and act control
                result = np.bitwise_xor(curr_control_vector,prev_control_vector)
                indexes , = np.where(result == 1)
                for index in indexes:
                    if index !=4:off_action = control_action_off[index]
                    if index !=4:on_action = control_action_on[index]

                    if index == 4 and curr_control_vector[index] == 1: #when space pressed means stop
                        pwm.set_pwm(0,0,servo_stop)
                        pwm.set_pwm(1,0,servo_mid)
                    elif curr_control_vector[index] == 0: #means release the key on the curr index
                        pwm.set_pwm(off_action[0],0,off_action[1])
                    else:                               #means press the key on the curr index
                        pwm.set_pwm(on_action[0],0,on_action[1])
            else: #there is no key pressed car stop
                print("no key")
                pwm.set_pwm(0,0,servo_stop)
                pwm.set_pwm(1,0,servo_mid)

            #update prev_control_vector
            prev_control_vector = curr_control_vector

            #stamp the current image with curr_control_vector
            label = ''.join(map(lambda x: str(x),curr_control_vector)) #make a string out of curr_control_vector
            count = path + str(index) + "_" + label + '.jpg'  #~/Desktop/index_command.jpg
            camera.capture(count,use_video_port=True)
            index += 1


    client_socket.close()

'''

            if data == "i": #if i goes foward
                pwm.set_pwm(0, 0, servo_forward)

            if data == "k": #if k goes backward
                pwm.set_pwm(0, 0, servo_back)

            if data == "j": #if j goes left
                pwm.set_pwm(1, 0, servo_left)

            if data == "l": #if l goes right
                pwm.set_pwm(1, 0, servo_right)

            if data == "f": #middle direction whille going forward
                pwm.set_pwm(1, 0, servo_mid)
                pwm.set_pwm(0, 0, servo_forward)

            if data == "b": #middle direction whille going backward
                pwm.set_pwm(1, 0, servo_mid)
                pwm.set_pwm(0, 0, servo_back)

            if data == " ": #if /s stop the car
                pwm.set_pwm(0, 0, servo_stop)
                pwm.set_pwm(1, 0, servo_mid)

            #there is command change, record the command the image with count
            count = path + str(index) + "_" + data + '.jpg'  #~/Desktop/index_command.jpg
            #camera.capture(count,use_video_port=True)
            index += 1

'''


if __name__ == "__main__":
    main()
