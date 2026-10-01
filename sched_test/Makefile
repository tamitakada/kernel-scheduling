test:
	nvcc -arch=native -o simple simple.cu
	LD_PRELOAD=./agent.so simple

sched:
	gcc -o scheduler_daemon scheduler_daemon.c -lpthread -lrt

clean:
	rm scheduler_daemon simple
