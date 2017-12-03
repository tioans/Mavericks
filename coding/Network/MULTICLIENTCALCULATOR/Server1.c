/* A simple server in the internet domain using TCP
The port number is passed as an argument */

//accept() creates a new socket, and by forking a process that new server process is going to handle this socket. the original socket with the server is unnafected by this
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>

void error(const char *msg)
{

perror(msg);
exit(1);
}

int main(int argc, char *argv[])
{
char operation;
int sockfd, newsockfd, portno;
socklen_t clilen;
char buffer[256];
struct sockaddr_in serv_addr, cli_addr;
int n;


if (argc < 2) 
{
fprintf(stderr,"ERROR, no port provided\n");
exit(1);
}

sockfd = socket(AF_INET, SOCK_STREAM, 0);

if (sockfd < 0)
error("ERROR opening socket");

bzero((char *) &serv_addr, sizeof(serv_addr));

portno = atoi(argv[1]);
serv_addr.sin_family = AF_INET;
serv_addr.sin_addr.s_addr = INADDR_ANY; 
serv_addr.sin_port = htons(portno);    

if (bind(sockfd, (struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0)
error("ERROR on binding");

listen(sockfd,5); 

clilen = sizeof(cli_addr);

do
{

newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);

if (newsockfd > 0)
{
 if(fork()==0)
{
int i,val1,val2,k=0,tmp,size_val2,result,ct,j;

n=write(newsockfd,"Connection initiated, write operation in aopb format:",53);
printf("Connection initiated!-server side\n");
fflush(stdout);

do
{
n=read(newsockfd,&val1,sizeof(int));

printf("%d\n",val1);
fflush(stdout);

n=read(newsockfd,&operation,sizeof(char));

printf("%c\n",operation);
fflush(stdout);

n=read(newsockfd,&val2,sizeof(int));

printf("%d\n",val2);
fflush(stdout);

switch(operation)
{
case '+':
result=val1+val2;
break;
case '-':
result=val1-val2;
break;
case '/':
result=val1/val2;
break;
case '*':
result=val1*val2;
break;
default:
printf("INCORRECT FORMAT\n");
}

printf("Here is the message: %d\n",result);
n = write(newsockfd,&result,sizeof(result));
}
while(operation!='E');

close(newsockfd);
}
}
else
error("Error accepting client!")'

}while(1);

close(sockfd);


return 0;
}
