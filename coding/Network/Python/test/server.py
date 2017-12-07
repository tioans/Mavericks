import socket
import os

(HOST,PORT) = ('10.8.0.2',2001)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1) 
conn, addr = s.accept()

with open('recv.wav','wb') as f:
  while True:
    l = conn.recv(1024)
    if not l: break
    f.write(l)

os.system("C:/Users/Mavericks/Documents/GitHub/Mavericks/coding/modelLoadPyV2/modelLoadPyV2/modelLoadPyV2.py 0")

s.close()