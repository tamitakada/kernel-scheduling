agent:
	gcc -shared -fPIC -o agent.so agent.c -lcupti -lcuda -lpthread -lrt \
		-I$$CUDA_ROOT/include -I$$CUDA_ROOT/extras/CUPTI/include -L$$CUDA_ROOT/extras/CUPTI/lib64

test:
	nvcc -arch=native -o simple simple.cu

sched:
	gcc -o scheduler_daemon scheduler_logging.c scheduler_daemon.c -lpthread -lrt

clean-sched:
	rm scheduler_daemon 
	
clean-simple:
	rm simple

clean-agent:
	rm agent.so

clean: clean-sched clean-simple clean-agent
