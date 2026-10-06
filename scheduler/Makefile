agent:
	g++ -std=c++20 -shared -fPIC -o agent.so agent.cpp \
		-I$$CUDA_ROOT/include -L$$CUDA_ROOT/lib64 \
		-lcuda -lcupti -lnvrtc -lpthread -lrt -ldl

simple:
	nvcc -arch=native -o simple simple.cu

sched:
	g++ -std=c++20 -o scheduler_daemon scheduler_daemon.cpp -lpthread -lrt

clean-sched:
	rm -f /dev/shm/kernel_sched_shm_v4
	rm scheduler_daemon 
	
clean-simple:
	rm simple

clean-agent:
	rm agent.so

clean: clean-sched clean-agent clean-simple