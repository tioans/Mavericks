#include <stdlib.h>
#include <iostream>
#include <pthread.h>
#include <semaphore.h>

using namespace std;

int *temp_prod,*temp_cons,first_time_prod=0,first_time_cons=0;
int prod_numb=4,cons_numb=4;
int multi_prod=1, multi_cons=1;
int prod_count=0,cons_count=0;
int box_buffer[10]={2,2,2,2,2,2,2,2,2,2};

sem_t prod_sem,cons_sem;
sem_t prod1,cons1;

void *producer1(void *smh)
{ 
 
 while(prod_count<prod_numb)
 { 
  sem_wait(&prod1);
  sem_wait(&prod_sem);

 if(first_time_prod==0)
  {
    temp_prod=box_buffer;
    first_time_prod=1;
  }
 else
  temp_prod++;
  
  if(*temp_prod==2)
  {
   *temp_prod=1;
   prod_count++;
   cout<<endl<<"Produced items count: "<<prod_count;
   cout<<" "<<"Value: "<<*temp_prod<<endl;
  }

  if(prod_count==10*multi_prod)
  {
    first_time_prod=0;
    multi_prod++;
  }

  sem_post(&cons_sem);
  sem_post(&prod1);
 }

}

void *producer2(void *smh)
{  
 while(prod_count<prod_numb)
 { 
  
  sem_wait(&prod1);
  sem_wait(&prod_sem);

 if(first_time_prod==0)
  {
    temp_prod=box_buffer;
    first_time_prod=1;
  }
 else
  temp_prod++;
  
  if(*temp_prod==2)
  {
   *temp_prod=1;
   prod_count++;
   cout<<endl<<"Produced items count: "<<prod_count;
   cout<<" "<<"Value: "<<*temp_prod<<endl;
  }
   
   if(prod_count==10*multi_prod)
  {
    first_time_prod=0;
    multi_prod++;
  }
  
  sem_post(&cons_sem);
  sem_post(&prod1);
 }
}

void *consumer1(void *smh)
{
 while(cons_count<cons_numb) 
 { 
  sem_wait(&cons1);
  sem_wait(&cons_sem);

  if(first_time_cons==0)
  {
    temp_cons=box_buffer;
    first_time_cons=1;
  }
  else 
    temp_cons++;
  
  if(*temp_cons==1)
  {
    *temp_cons=2;
    cons_count++;
    cout<<endl<<"Consumed items count: "<<cons_count;
    cout<<" "<<"Value: "<<*temp_cons<<endl;
  }

  if(cons_count==10*multi_cons)
  {
    first_time_cons=0;
    multi_cons++;
  }

  sem_post(&prod_sem);
  sem_post(&cons1);
 }

}

void *consumer2(void *smh)
{

 while(cons_count<cons_numb)
 { 
  sem_wait(&cons1);
  sem_wait(&cons_sem);
 
 if(first_time_cons==0)
  {
    temp_cons=box_buffer;
    first_time_cons=1;
  }
  else
    temp_cons++;

  if(*temp_cons==1)
  {
    *temp_cons=2;
    cons_count++;
    cout<<endl<<"Consumed items count: "<<cons_count;
    cout<<" "<<"Value: "<<*temp_cons<<endl;
  }
 
  if(cons_count==10*multi_cons)
  {
    first_time_cons=0;
    multi_cons++;
  }

  sem_post(&prod_sem);
  sem_post(&cons1);
 }
  
}

int main(int argc, char *argv[])
{

	pthread_t thread1,thread2,thread3,thread4;
	int check1,check2,check3,check4,chk1,chk2,chk3,chk4;

chk1=sem_init(&prod_sem,0,4);
chk2=sem_init(&cons_sem,0,0);
chk3=sem_init(&prod1,0,1);
chk4=sem_init(&cons1,0,1);

check1=pthread_create(&thread1,NULL,producer1,NULL);
check2=pthread_create(&thread2,NULL,consumer1,NULL);
check3=pthread_create(&thread3,NULL,producer2,NULL);
check4=pthread_create(&thread4,NULL,consumer2,NULL);

if(check1!=0)
	cout<<endl<<"Error on thread1 creation!"<<endl;
if(check2!=0)
	cout<<endl<<"Error on thread2 creation!"<<endl;
if(check3!=0)
  cout<<endl<<"Error on thread3 creation!"<<endl;
if(check4!=0)
  cout<<endl<<"Error on thread4 creation!"<<endl;

if(chk1!=0)
  cout<<endl<<"Error on c1 creation!"<<endl;
if(chk2!=0)
  cout<<endl<<"Error on c2 creation!"<<endl;
if(chk3!=0)
  cout<<endl<<"Error on c3 creation!"<<endl;
if(chk4!=0)
  cout<<endl<<"Error on c4 creation!"<<endl;

pthread_join(thread4,NULL);
pthread_join(thread2,NULL);
pthread_join(thread1,NULL);
pthread_join(thread3,NULL);

cout<<endl<<"Program ended!"<<endl;

return 0;
}

