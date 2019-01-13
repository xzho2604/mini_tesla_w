import socket
import RPi.GPIO as GPIO
import time
# Import the PCA9685 module.
import Adafruit_PCA9685


# Initialise the PCA9685 using the default address (0x40).
pwm = Adafruit_PCA9685.PCA9685()

servo_stop = 360   # Min pulse length out of 4096
servo_forward = 390 # Max pulse length out of 4096
servo_back = 340

servo_mid = 400
servo_left = 460
servo_right = 350

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

def main():
    #create tcp socket
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.bind(("",5555))
    tcp_socket.listen()

    client_socket,client_addr = tcp_socket.accept()
    while True:
        data = client_socket.recv(4096)
        data = data.decode('UTF-8')
        if data: #when there is command
            print(data) #if there is command print command
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

            if data == "b": #middle direction whille going forward
                pwm.set_pwm(1, 0, servo_mid)
                pwm.set_pwm(0, 0, servo_back)

            if data == " ": #middle direction whille going forward
                pwm.set_pwm(1, 0, servo_mid)
                pwm.set_pwm(1, 0, servo_back)

    client_socket.close()

if __name__ == "__main__":
    main()
    
