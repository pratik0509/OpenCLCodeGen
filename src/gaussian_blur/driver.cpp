#include <iostream>
#include<bits/stdc++.h>
#include <vector>
#include <string>
#include <CL/cl.hpp>

#define SIZE 50000

int main(){
    //get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(all_platforms.size()==0){
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform=all_platforms[1];
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";
    
    //get default device of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Device default_device=all_devices[0];
    std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";
    
    
    cl::Context context({default_device});
    
    cl::Program::Sources sources;
    
    std::vector<char> kcode;
    std::ifstream file("kernel.cl", std::ios::in|std::ios::binary|std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    file.read(kcode.data(), SIZE);
    file.close();
    std::string kernel_code(kcode.begin(), kcode.end());
    sources.push_back({kernel_code.c_str(),kernel_code.length()});
    
    cl::Program program(context,sources);
    std::vector<cl::Device> devs;
    devs.push_back(default_device);
    if(program.build(devs)!=CL_SUCCESS){
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<<"\n";
        exit(1);
    }
    
    const int N = 32;
    const int M = 32;
    const int l = 3;
    const int m = 3;
    // create buffers on the device
    cl::Buffer buffer_img(context,CL_MEM_READ_ONLY,sizeof(float)*N*M);
    cl::Buffer buffer_fil(context,CL_MEM_READ_ONLY,sizeof(float)*l*m);
    cl::Buffer buffer_out(context,CL_MEM_READ_WRITE,sizeof(float)*N*M);
    cl::Buffer buffer_N(context,CL_MEM_READ_WRITE,sizeof(int));
    cl::Buffer buffer_M(context,CL_MEM_READ_WRITE,sizeof(int));
    cl::Buffer buffer_l(context,CL_MEM_READ_WRITE,sizeof(int));
    cl::Buffer buffer_m(context,CL_MEM_READ_WRITE,sizeof(int));
    
    float img[N*M], fil[l*m], out[N*M];
    for(int i = 0; i < N*M; ++i) img[i] = i+1;
    for(int i = 0; i < l*m; ++i) fil[i] = i+1;
    //create queue to which we will push commands for the device.
    cl::CommandQueue queue(context,default_device);
    

    queue.enqueueWriteBuffer(buffer_img,CL_TRUE,0,sizeof(float)*N*M,img);
    queue.enqueueWriteBuffer(buffer_fil,CL_TRUE,0,sizeof(float)*l*l,fil);
    //queue.enqueueWriteBuffer(buffer_out,CL_TRUE,0,sizeof(float)*N*M,out);
    queue.enqueueWriteBuffer(buffer_N,CL_TRUE,0,sizeof(int),&N);
    queue.enqueueWriteBuffer(buffer_M,CL_TRUE,0,sizeof(int),&M);
    queue.enqueueWriteBuffer(buffer_l,CL_TRUE,0,sizeof(int),&l);
    queue.enqueueWriteBuffer(buffer_m,CL_TRUE,0,sizeof(int),&m);
    
    
    //run the kernel
    //cl::KernelFunctor simple_add(cl::Kernel(program,"krnel"),queue,cl::NullRange,cl::NDRange(N,M),cl::NullRange);
    //simple_add(buffer_img,buffer_fil,buffer_out, buffer_N, buffer_M, buffer_l, buffer_m);
    
    //alternative way to run the kernel
    cl::Kernel kernel_add=cl::Kernel(program,"krnel");
    kernel_add.setArg(0,buffer_img);
    kernel_add.setArg(1,buffer_fil);
    kernel_add.setArg(2,buffer_out);
    kernel_add.setArg(3,buffer_N);
    kernel_add.setArg(4,buffer_M);
    kernel_add.setArg(5,buffer_l);
    kernel_add.setArg(6,buffer_m);
    queue.enqueueNDRangeKernel(kernel_add,cl::NullRange,cl::NDRange(N, M),cl::NullRange);
    if(queue.finish() == CL_SUCCESS) {
	    std::cout << "Correctly executed\n";
    };
    
    // int C[10];
    //read result C from the device to array C
    queue.enqueueReadBuffer(buffer_out, CL_TRUE,0,sizeof(float)*N*M, out);
for (int i = 0; i < 25; ++i) 
    std::cout<<" result: \n" << out[i] << "\n";
    
    return 0;
}
