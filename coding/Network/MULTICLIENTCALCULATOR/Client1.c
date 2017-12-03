#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>

void error(const char *msg)
{
perror(msg);
exit(0);
}

int main(int argc, char *argv[])
{
int sockfd, portno, n, val1,val2,result;
struct sockaddr_in serv_addr;
struct hostent *server;
char buffer[256], operation;

if (argc < 3) 
{
fprintf(stderr,"usage %s hostname port\n", argv[0]);
exit(0);
}

portno = atoi(argv[2]);
//creates socketfd for client
sockfd = socket(AF_INET, SOCK_STREAM, 0);
if (sockfd < 0)
error("ERROR opening socket");
//obsolete function, returns a structure of type hostent for the given host name
server = gethostbyname(argv[1]);
if (server == NULL) 
{
fprintf(stderr,"ERROR, no such host\n");
exit(0);
}

bzero((char *) &serv_addr, sizeof(serv_addr));
serv_addr.sin_family = AF_INET;
//bcopy(const void *src, void *dest, size_t n);
//copies n bytes from src to dest
bcopy((char *)server->h_addr,(char *)&serv_addr.sin_addr.s_addr,server->h_length);

//printf("step1c");

serv_addr.sin_port = htons(portno);
if (connect(sockfd,(struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0)
error("ERROR connecting");

n = read(sockfd,buffer,255);
//if (n < 0)
//error("ERROR reading from socket");

printf("%s",buffer);
fflush(stdout);

do
{

scanf("%s",buffer);
sscanf(buffer,"%d%c%d",&val1,&operation,&val2);

bzero(buffer,256);

n = write(sockfd,&val1,sizeof(int));

n = write(sockfd,&operation,sizeof(char));

n = write(sockfd,&val2,sizeof(int));
/*
printf("val1=%d",val1);
fflush(stdout);
printf("operation=%c",operation);
fflush(stdout);
printf("val2=%d\n",val2);
fflush(stdout);
*/
n = read(sockfd,&result,sizeof(int));
if (n < 0)
error("ERROR reading from socket");

if(operation!='E')
{
printf("Answer from server: ");

printf("%d",result);

printf("\nPlease enter the operation (aopb format): ");

fflush(stdout);
}
}while(operation!='E');


close(sockfd);


return 0;
}


