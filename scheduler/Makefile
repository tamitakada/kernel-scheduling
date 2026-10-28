agent:
	g++ -std=c++20 -shared -fPIC -o agent.so agent.cpp \
		-I$$CUDA_ROOT/include -I$$CUDA_ROOT/extras/CUPTI/include -L$$CUDA_ROOT/lib64 -L$$CUDA_ROOT/extras/CUPTI/lib64 -L$$CUDA_ROOT/targets \
		-lcuda -lcupti -lnvrtc -lpthread -lrt -ldl

simple:
	nvcc -arch=native -o simple simple.cu

sched:
	g++ -std=c++20 -o scheduler_daemon scheduler_daemon.cpp -lpthread -lrt

pairprof:
	g++ -std=c++20 -o pairprof_daemon pairprof_daemon.cpp -lpthread -lrt

clean-pair:
	rm -f /dev/shm/kernel_sched_shm_*
	rm pairprof_daemon

clean-sched:
	rm -f /dev/shm/kernel_sched_shm_*
	rm scheduler_daemon 
	
clean-simple:
	rm simple

clean-agent:
	rm agent.so

clean: clean-sched clean-agent clean-simple
