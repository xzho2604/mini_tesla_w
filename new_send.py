import time
import socket
import pygame

def main():
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #the receiver address
    tcp_ip = '192.168.1.12'
    tcp_port = 5555
    tcp_socket.connect((tcp_ip, tcp_port))
#    tcp_socket.setblocking(0)
    
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
    flag =0 #as control change status
    status = '' #current key board status

    while(True):
        events = pygame.event.get()
        #check if there is event if no then flag = 0
        if not events:  
            flag = 0
        for event in events: #goes into the loop when only there is event
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
                    #tcp_socket.send(mes.encode())
                    time.sleep(0.1)
                    flag = 1
                    status += mes
                    
                elif event.type == pygame.KEYUP:
                    if((pygame.key.get_pressed()[pygame.K_i] != 0 and pygame.key.get_pressed()[pygame.K_j] != 0) or (pygame.key.get_pressed()[pygame.K_i] !=0 and pygame.key.get_pressed()[pygame.K_l] != 0)):
                        mes = mes					
                    elif((pygame.key.get_pressed()[pygame.K_i] != 0 and pygame.key.get_pressed()[pygame.K_j] != 0) or (pygame.key.get_pressed()[pygame.K_k] !=0 and pygame.key.get_pressed()[pygame.K_l] != 0)):
                         mes = mes					
                    elif(pygame.key.get_pressed()[pygame.K_i] != 0): #means the right or left turn released while going forward so direction middle
                        mes = "f" 
                        status = 'i'
                    elif(pygame.key.get_pressed()[pygame.K_k] != 0): #means right or left turn relased while going back so direction middle
                        mes = "b" 
                        status = 'k'
                    elif(pygame.key.get_pressed()[pygame.K_j] != 0): #only goes left 
                        mes = "j"
                        status = 'j'
                    elif(pygame.key.get_pressed()[pygame.K_d] != 0): #only goes right
                        mes = "l"
                        status = 'l'
                    else: #means stop
                        mes = " "
                        status = 'n' 
                    #tcp_socket.send(mes.encode())
                    time.sleep(0.1)
                    flag = 1
                else:  #there is event but not keydown or up flag = 0
                    flag = 0
       
        #now we check if there is command change
        if flag == 0: #message indicating nothing happens
            #tcp_socket.send("n".encode())
            time.sleep(0.1)

        tcp_socket.send(status.encode())
        print(status)

            
    tcp_socket.close()

if __name__ == "__main__":
    main()

