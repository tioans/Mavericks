import socket
import shutil
import sys
import subprocess
import multiprocessing as mp
import os

class raiseExcept(Exception):
    pass

def Server(server_ip,port,numb_of_clients):

    (HOST,PORT) = (server_ip,port)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(numb_of_clients)
    return s

def clientAccept(servSock):

    print("Server Initiated! \nReady to receive Client!")
    conn, addr = servSock.accept()
    return conn, addr

def serverOps(connSock,servSock,model_name,train_option):
    
    recv_file_name = 'sound.wav'
    test_folder = 'C:/Users/Mavericks/Documents/Simple_LSTM/RNN-Implementation/data/raw/librivox/LibriSpeech/test-clean-wav'

    with open(recv_file_name,'wb') as f:
        while True:
            l=connSock.recv(1024)
            if not l:
                break
            f.write(l)
    dest = os.path.join(test_folder,recv_file_name)
    shutil.move(recv_file_name,dest)

    p = subprocess.Popen(['python', 'C:/Users/Mavericks/Documents/GitHub/Mavericks/coding/modelLoadPyV2/modelLoadPyV2/modelLoadPyV2.py', train_option, model_name, 'S117'], 
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding='utf8')

    decoded = p.stdout.readline().strip()
    decoded_bytes = str.encode(decoded)
    connSock.sendall(decoded_bytes)

    servSock.close()

if __name__ == "__main__":
    
    try: 
        numb_of_args=len(sys.argv)-1

        if not numb_of_args == 4:           # comment these two lines test in IDE
            raise raiseExcept()
            
    except(raiseExcept):
        print("Number of input parameters wrong! Correct format: 'ip' 'port'")
        sys.exit()

    try:
        socket.inet_aton(sys.argv[1])
        #socket.inet_aton('10.8.0.2')        # uncomment this and comment line above to test in IDE
        print ("IPv4 address, port:")

    except socket.error:
        print ("Not IPv4 address!")
        sys.exit()
     
    try:
        port = sys.argv[2]
        #port = 2001                         # uncomment this and comment line above to test in IDE
        port = int(port)
        train_option = sys.argv[4]
        print(sys.argv[1],port)
        numb_of_clients = 1
        servSock = Server(sys.argv[1],port,numb_of_clients)
        #servSock = Server('10.8.0.2',port,numb_of_clients)   # uncomment this and comment line above to test in IDE
        
        #jobs=[]
        #while True:
        #    connSock, _ = clientAccept(servSock)           
        #    p = mp.Process(target = serverOps, args =(connSock,servSock,sys.argv[3],train_option))
        #    jobs.append(p)
        #    p.start()

        connSock, _ = clientAccept(servSock)  
        serverOps(connSock,servSock,sys.argv[3],train_option)    
          

    except(raiseExcept): 
        print("Error while initializng socket!")


##################################################Multi input code########################################################
#txt_to_send = 'sound.txt'
#file_location = os.path.join(test_folder,txt_to_send)


#temp_bytes = conn.recv(152493)
#temp = temp_bytes.decode("utf-8")
#print(temp)

#while temp == 'y' or temp == 'Y':
    
#    with open(recv_file_name,'wb') as f:
#      while True:
#        l = conn.recv(1024)
#        if not l: 
#            break
#        f.write(l)

#    shutil.copy(recv_file_name,test_folder)
    
#    temp = temp + '\n'
#    p.stdin.write(temp)
#    p.stdin.flush()

#    decoded = p.stdout.readline().strip()
#    decoded_bytes = str.encode(decoded)
#    conn.sendall(decoded_bytes)
#    conn.shutdown(socket.SHUT_WR)

#    temp_bytes = conn.recv(1024)
#    temp = temp_bytes.decode("utf-8")
######################################################END#################################################################