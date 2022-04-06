nvcc -forward-unknown-to-host-compiler  -I/usr/local/cuda/include -I/root/sputnik -I/root/libcusparse_lt/include -I/root/sputnik/third_party/abseil-cpp -L/usr/local/cuda/lib64  -L/root/sputnik/build/sputnik -L/root/libcusparse_lt/lib64  -lcusparse -lcudart -lcusparseLt -lspmm  --generate-code=arch=compute_80,code=sm_80 -std=c++14 -g -G spmma_example.cu -o spmm
