/**
Each kernel computes the output corresponding to one pixel in
the final image. The global_id of the thread corresponds to the
top left cell in image matrix.

NOT TESTED
 */

__kernel void kernel(const __global float *input, const __global float *filter,
 const __global float *output, int imgHeight, int imgWidth, int filHeight, int filWidth) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    float output_val = 0;
    for (int i = 0; i < filHeight; ++i)
        for (int j = 0; j < filWidth; ++j)
            output += input[row*imgWidth + col] * filter[i*filWidth + j];
    output[row*(imgWidth - filWidth + 1) + col] = output_val;
}