    #include <iostream>
    #include<bits/stdc++.h>
    #include <vector>
    #include <string>
    #include <CL/cl.hpp>
     
    int main(){
        //get all platforms (drivers)
        std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
        if(all_platforms.size()==0){
            std::cout<<" No platforms found. Check OpenCL installation!\n";
            exit(1);
        }
        cl::Platform default_platform=all_platforms[0];
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
     
        // kernel calculates for each element C=A+B
        /*
        std::string kernel_code=
                "   void kernel simple_add(global const int* A, global const int* B, global int* C){       "
                "       C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];                 "
                "   }                                                                               ";
        */
        std::string kernel_code = "                __kernel "
                "void matrix_trace(__global int* ARR, __global int* S, __global int* N, __global int* B) {"
                 "   int temp;"
                    "int n_index = get_global_id(0);"
                    "int t = N[0];"
                    "int i;ARR[n_index] = 1;"
                    "while (t > 0) {"
                        "temp = 0;"
                        "for (i = n_index * B[0]; i < N[0]; ++i) {"
                            "temp += ARR[i];"
                        "}"
                        "ARR[n_index/B[0]] = temp;"
                        "t /= B[0];"
                        "barrier(CLK_LOCAL_MEM_FENCE);"
                        //"if (n_index % B[0])"
                          //  "break;"
                    "}"
		    "if (n_index == 2) S[0] = ARR[0];"
                    "return;"
                "}"
        ;
//	std::cout << kernel_code << std::endl;
        sources.push_back({kernel_code.c_str(),kernel_code.length()});
     
        cl::Program program(context,sources);
	std::vector<cl::Device> devs;
	devs.push_back(default_device);
        if(program.build(devs)!=CL_SUCCESS){
            std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<<"\n";
            exit(1);
        }
     
     
        // create buffers on the device
        cl::Buffer buffer_ARR(context,CL_MEM_READ_WRITE,sizeof(int)*10);
        cl::Buffer buffer_S(context,CL_MEM_READ_WRITE,sizeof(int));
        cl::Buffer buffer_N(context,CL_MEM_READ_WRITE,sizeof(int));
        cl::Buffer buffer_B(context,CL_MEM_READ_WRITE,sizeof(int));
     
        int A[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
     
        //create queue to which we will push commands for the device.
        cl::CommandQueue queue(context,default_device);
     

        int S = 0;
        int N = 10;
        int B = 2;
        queue.enqueueWriteBuffer(buffer_ARR,CL_TRUE,0,sizeof(int)*10,A);
        queue.enqueueWriteBuffer(buffer_S,CL_TRUE,0,sizeof(int),&S);
        queue.enqueueWriteBuffer(buffer_N,CL_TRUE,0,sizeof(int),&N);
        queue.enqueueWriteBuffer(buffer_B,CL_TRUE,0,sizeof(int),&B);
     
     
        //run the kernel
    //    cl::KernelFunctor simple_add(cl::Kernel(program,"simple_add"),queue,cl::NullRange,cl::NDRange(10),cl::NullRange);
      //  simple_add(buffer_A,buffer_B,buffer_C);
     
        //alternative way to run the kernel
        cl::Kernel kernel_add=cl::Kernel(program,"matrix_trace");
        kernel_add.setArg(0,buffer_ARR);
        kernel_add.setArg(1,buffer_S);
        kernel_add.setArg(2,buffer_N);
        kernel_add.setArg(3,buffer_B);
        queue.enqueueNDRangeKernel(kernel_add,cl::NullRange,cl::NDRange(10/B),cl::NullRange);
        queue.finish();
     
        // int C[10];
        //read result C from the device to array C
        queue.enqueueReadBuffer(buffer_ARR,CL_TRUE,0,sizeof(int)*10,A);
	for (int i = 0; i < 10; ++i) 
        std::cout<<" result: \n" << A[i] << "\n";
     
        return 0;
    }
