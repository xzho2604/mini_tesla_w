#self driving by the intruction sent by the pc computed by the model
from collections import OrderedDict
import numpy as np
import socket
import RPi.GPIO as GPIO
import time
import datetime
import Adafruit_PCA9685
import picamera


# Initialise the PCA9685 using the default address (0x40).
pwm = Adafruit_PCA9685.PCA9685()

servo_stop = 360   # Min pulse length out of 4096
servo_forward = 390 # Max pulse length out of 4096
servo_back = 335

servo_mid = 400
servo_left = 470
servo_right = 340

# Set frequency to 60hz, good for servos.
pwm.set_pwm_freq(60)
#=====================================================================
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

#process the comming data from the socket remove the duplicate and turn string command into list
def process_data(data): #data in the form of 110101
    if len(data) > 5: return False
    command = [int(x) for x in list(data)]
    return command



#=====================================================================

#=====================================================================
#listen to the comand from the model
def drive():
    print('Start self driving mode')
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.bind(("",8888))   #port 8888 is used for auto pilot
    tcp_socket.listen()
    client_socket,client_addr = tcp_socket.accept()

    #init the control dict and vector
    control_dict = {'i':0,'k':1,'l':2,'j':3,' ':4}
    prev_control_vector = np.array([0,0,0,0,0])
    #control index:[channel,servo_val] key at index pressed
    control_action_on = {0:[0,servo_forward], 1:[0,servo_back], 2:[1,servo_right], 3:[1,servo_left]}
    control_action_off = {0:[0,servo_stop], 1:[0,servo_stop], 2:[1,servo_mid], 3:[1,servo_mid]}

    while True:
        data = client_socket.recv(4096)
        data = data.decode('UTF-8') #data string in the form of 110011

        #remove the duplicate transmission and put into list of command
        curr_control_vector= process_data(data)
        if curr_control_vector: #if there is no duplicate
            curr_control_vector = np.array(curr_control_vector)
            #print(curr_control_vector)
            #now we have the command list we can issue the comand to the car
            result = np.bitwise_xor(curr_control_vector,prev_control_vector)
            indexes , = np.where(result == 1)
            for index in indexes:
                if index !=4:off_action = control_action_off[index]
                if index !=4:on_action = control_action_on[index]

                if index == 4 and curr_control_vector[index] == 1: #when space pressed means stop
                    print("stop car")
                    pwm.set_pwm(0,0,servo_stop)
                    pwm.set_pwm(1,0,servo_mid)
                elif curr_control_vector[index] == 0: #means release the key on the curr index
                    print("channle:", off_action[0],"value:", off_action[1])
                    pwm.set_pwm(off_action[0],0,off_action[1])
                else:                               #means press the key on the curr index
                    print("channle:", on_action[0],"value:", on_action[1])
                    pwm.set_pwm(on_action[0],0,on_action[1])

            #update prev_control_vector
            prev_control_vector = curr_control_vector
    client_socket.close()

#=====================================================================
drive()
