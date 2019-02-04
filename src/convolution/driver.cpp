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
    const int m = 9;

    int input[n], output[n], filter[m];     // A is initial, B is result, C is expected result
    for (int i=0; i<n; i++)
        input[i] = i;
    for (int i = 1; i <= m; ++i)
        filter[i-1] = i;
    Platform default_platform = getPlatform();
    Device default_device     = getDevice(default_platform, 0);
    Context context({default_device});
    Program::Sources sources;

    std::ifstream t("kernel.cl");
    std::string kernel_code((std::istreambuf_iterator<char>(t)),
                 std::istreambuf_iterator<char>());
    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    Program program(context, sources);
    if (program.build({default_device}) != CL_SUCCESS) {
        cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
        exit(1);
    }
    
    Buffer buffer_inp(context, CL_MEM_READ_WRITE, sizeof(int) * n);
    Buffer buffer_out(context, CL_MEM_READ_WRITE, sizeof(int) * n);
    Buffer buffer_fil(context, CL_MEM_READ_WRITE, sizeof(int) * m);
    Buffer buffer_sz(context, CL_MEM_READ_WRITE, sizeof(int));
    CommandQueue queue(context, default_device);
    int k = m;
    queue.enqueueWriteBuffer(buffer_inp, CL_TRUE, 0, sizeof(int)*n, input);
    queue.enqueueWriteBuffer(buffer_out, CL_TRUE, 0, sizeof(int)*n, output);
    queue.enqueueWriteBuffer(buffer_fil, CL_TRUE, 0, sizeof(int)*m, filter);

    Kernel convolve = Kernel(program, "filter");
    convolve.setArg(0, buffer_inp);
    convolve.setArg(1, buffer_out);
    convolve.setArg(2, buffer_fil);
    convolve.setArg(3, sizeof(int), &k);

    queue.enqueueNDRangeKernel(convolve, NullRange, NDRange(n), NullRange);
    queue.finish();
    queue.enqueueReadBuffer(buffer_out, CL_TRUE, 0, sizeof(int)*n, output);
    
    for(int i = 0; i < 20; i++)
    {
        cout << output[i] << " ";
    }
    return 0;
}

