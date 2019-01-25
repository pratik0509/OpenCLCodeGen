#include <iostream>
#include <algorithm>
#include <iterator>
#include <stdio.h>
#include <fstream>
#include <CL/cl.hpp>
#include <string>
#include <streambuf>

using namespace std;
using namespace cl;


#define SIZE 50000

Platform getPlatform() {
    /* Returns the first platform found. */
    std::vector<Platform> all_platforms;
    Platform::get(&all_platforms);

    if (all_platforms.size()==0) {
        cout << "No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    return all_platforms[0];
}


Device getDevice(Platform platform, int i, bool display=false) {
    /* Returns the deviced specified by the index i on platform.
     * If display is true, then all of the platforms are listed.
     */
    std::vector<Device> all_devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        cout << "No devices found. Check OpenCL installation!\n";
        exit(1);
    }

    if (display) {
        for (int j=0; j<all_devices.size(); j++)
            printf("Device %d: %s\n", j, all_devices[j].getInfo<CL_DEVICE_NAME>().c_str());
    }
    return all_devices[i];
}


int main() {
    const int n = 1024;    // size of vectors
    const int c_max = 5;   // max value to iterate to

    int A[n], B[n], C[n];     // A is initial, B is result, C is expected result
    for (int i=0; i<n; i++) {
        A[i] = i;
        C[i] = i*i;
    }
    Platform default_platform = getPlatform();
    Device default_device     = getDevice(default_platform, 0);
    Context context({default_device});
    Program::Sources sources;

    std::ifstream t("kernel.cl");
    std::string kernel_code((std::istreambuf_iterator<char>(t)),
                 std::istreambuf_iterator<char>());
    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    // cout << kernel_code.c_str();
    Program program(context, sources);
    if (program.build({default_device}) != CL_SUCCESS) {
        cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
        exit(1);
    }
    
    Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * n);
    Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * n);
    CommandQueue queue(context, default_device);
    queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int)*n, A);
    // queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int)*n, B);

    Kernel multiply_by = Kernel(program, "filter");
    multiply_by.setArg(0, buffer_A);
    multiply_by.setArg(1, buffer_B);

    // for (int c=2; c<=c_max; c++) {
    //     multiply_by.setArg(1, c);
    // }
    queue.enqueueNDRangeKernel(multiply_by, NullRange, NDRange(n), NullRange);
    queue.finish();
    queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, sizeof(int)*n, B);
    
    for(int i = 0; i < 20; i++)
    {
        cout << B[i] << " ";
    }
    if (std::equal(std::begin(B), std::end(B), std::begin(C)))
    
        cout << "Arrays are equal!" << endl;
    else
        cout << "Uh-oh, the arrays aren't equal!" << endl;

    return 0;
}

