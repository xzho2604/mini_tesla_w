import time
import socket
import pygame

def main():
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #the receiver address
    tcp_ip = '192.168.1.15'
    tcp_port = 5555
    tcp_socket.connect((tcp_ip, tcp_port))
    
    #set up the pygame
    pygame.init()
    screen = pygame.display.set_mode((240, 240))
    pygame.display.set_caption('Pi Car')

    print("w/s: acceleration")
    print("a/d: steering")
    print("UP/DOWN: tilt")
    print("LEFT/RIGHT: pan")
    print("esc: exit")
    
    mes = ""
    while(True):
        for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    keys = pygame.key.get_pressed()

                    if keys[pygame.K_i]:
                        mes = "i"
                    if keys[pygame.K_k]:
                        mes = "k"
                    if keys[pygame.K_j]:
                        mes = "j"
                    if keys[pygame.K_l]:
                        mes = "l"
                    if keys[pygame.K_SPACE]:
                        mes = " "
                    tcp_socket.send(mes.encode())
                    time.sleep(0.1)
                    
                elif event.type == pygame.KEYUP:
                    if((pygame.key.get_pressed()[pygame.K_i] != 0 and pygame.key.get_pressed()[pygame.K_j] != 0) or (pygame.key.get_pressed()[pygame.K_i] !=0 and pygame.key.get_pressed()[pygame.K_l] != 0)):
                        mes = mes					
                    elif((pygame.key.get_pressed()[pygame.K_i] != 0 and pygame.key.get_pressed()[pygame.K_j] != 0) or (pygame.key.get_pressed()[pygame.K_k] !=0 and pygame.key.get_pressed()[pygame.K_l] != 0)):
                         mes = mes					
                    elif(pygame.key.get_pressed()[pygame.K_i] != 0): #means the right or left turn released while going forward so direction middle
                        mes = "f" 
                    elif(pygame.key.get_pressed()[pygame.K_k] != 0): #means right or left turn relased while going back so direction middle
                        mes = "b" 
                    elif(pygame.key.get_pressed()[pygame.K_j] != 0): #only goes left 
                        mes = "j"
                    elif(pygame.key.get_pressed()[pygame.K_d] != 0): #only goes right
                        mes = "l"
                    else: #means stop
                        mes = " "
                    tcp_socket.send(mes.encode())
                    time.sleep(0.1)


            
    tcp_socket.close()

if __name__ == "__main__":
    main()

