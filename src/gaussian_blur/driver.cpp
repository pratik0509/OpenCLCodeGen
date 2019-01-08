#include <bits/stdc++.h>
#include <CL/cl.hpp>

#define N 32
#define M 32
#define l 3
#define m 3
#define SIZE 200009
#define NUM_RUNS 1

char *kernelStr;

int main() {
    float *img = (float*)malloc(N*M*sizeof(float));
    float *outimg = (float*)malloc(N*M*sizeof(float));
    float *fil = (float*)malloc(l*m*sizeof(float));
    img[0] = img[1] = img[2] = img[3] = 1;
    fil[0] = fil[1] = fil[2] = fil[3] = 1;
    kernelStr = (char*)malloc(SIZE * sizeof(char));
    // Configure the OpenCL environment
    printf(">>> Initializing OpenCL...\n");
    cl_platform_id platform = 0;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device = 0;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);
    char deviceName[1024];
    clGetDeviceInfo(device, CL_DEVICE_NAME, 1024, deviceName, NULL);
    cl_event event = NULL;

    // Compile the kernel
    std::ifstream file("kernel.cl", std::ios::in|std::ios::binary|std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    file.read(kernelStr, SIZE);
    file.close();
    std::cout << kernelStr;
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelStr, NULL, NULL);
    clBuildProgram(program, 0, NULL, "", NULL, NULL);

    // Check for compilation errors
    size_t logSize;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    char* messages = (char*)malloc((1+logSize)*sizeof(char));
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, messages, NULL);
    messages[logSize] = '\0';
    if (logSize > 10) { printf(">>> Compiler message: %s\n", messages); }
    free(messages);

    // Prepare OpenCL memory objects
    cl_mem bufImg = clCreateBuffer(context, CL_MEM_READ_ONLY,  M*N*sizeof(float), NULL, NULL);
    cl_mem bufFil = clCreateBuffer(context, CL_MEM_READ_ONLY,  l*m*sizeof(float), NULL, NULL);
    cl_mem bufOut = clCreateBuffer(context, CL_MEM_READ_WRITE, M*N*sizeof(float), NULL, NULL);

    // Copy matrices to the GPU
    clEnqueueWriteBuffer(queue, bufImg, CL_TRUE, 0, M*N*sizeof(float), img, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufFil, CL_TRUE, 0, l*m*sizeof(float), fil, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufOut, CL_TRUE, 0, M*N*sizeof(float), outimg, 0, NULL, NULL);

    int imgH = M, imgW = N, filH = l, filW = m;
    // Configure the myGEMM kernel and set its arguments
    cl_kernel kernel = clCreateKernel(program, "convolution", NULL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&bufImg);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&bufFil);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&bufOut);
    clSetKernelArg(kernel, 3, sizeof(int), (void*)&imgH);
    clSetKernelArg(kernel, 4, sizeof(int), (void*)&imgW);
    clSetKernelArg(kernel, 5, sizeof(int), (void*)&filH);
    clSetKernelArg(kernel, 6, sizeof(int), (void*)&filW);

    // Start the timed loop
    printf(">>> Starting %d runs...\n", NUM_RUNS);
    for (int r=0; r<NUM_RUNS; r++) {

        // Run the myGEMM kernel
        const size_t local[2] = { 1, 1 };
        const size_t global[2] = { M, N };
        clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, &event);

        // Wait for calculations to be finished
        clWaitForEvents(1, &event);
    }


    // Copy the output matrix C back to the CPU memory
    clEnqueueReadBuffer(queue, bufOut, CL_TRUE, 0, M*N*sizeof(float), outimg, NULL, NULL, NULL);

    // Output the Result
    for (int i = 0; i < M; ++i) {
	for (int j = 0; j < N; ++j) {
	    std::cout << outimg[N*i + j] << " ";
	}
    std::cout << std::endl;
	}
    // Free the OpenCL memory objects
    clReleaseMemObject(bufImg);
    clReleaseMemObject(bufFil);
    clReleaseMemObject(bufOut);

    // Clean-up OpenCL 
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseProgram(program);
    clReleaseKernel(kernel);

    // Free the host memory objects
    free(img);
    free(outimg);
    free(fil);

    // Exit
    return 0;
}
