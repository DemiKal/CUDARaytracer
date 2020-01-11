



__global__ void addKernel(int* c, const int* a, const int* b);
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);
