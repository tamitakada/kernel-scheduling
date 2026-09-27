sched:
	gcc -o sched sched.c -lpthread -lrt

cupti:
	gcc -shared -fPIC -o local_sched.so local_sched.c -lcupti -lcuda -lpthread -lrt -I$CUDA_ROOT/include -I$CUDA_ROOT/extras/CUPTI/include -L$CUDA_ROOT/extras/CUPTI/lib64

clean:
	rm *.so
	rm sched
