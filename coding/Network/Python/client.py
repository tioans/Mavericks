import socket

(HOST,PORT)=('10.8.0.3',19123)
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.connect((HOST,PORT))

with open('microphone-results.wav', 'rb') as f:
  for l in f: s.sendall(l)

s.shutdown(socket.SHUT_WR)

with open('recv.wav','wb') as f:
  while True:
    l = s.recv(1024)
    if not l: break
    f.write(l)

s.close()