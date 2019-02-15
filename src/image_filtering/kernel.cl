void kernel filter(global int* input_img, global int* output_img,
    global int* filter, int N, int M, int r, int c) {
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    int C = M - (c - 1);
    output_img[x*M + y] = 0;
    if (x + r/2 >= N || y + c/2 >= M)
        return; 
    if (x < r/2 || y < c/2)
        return; 
        
    for (uint i = 0; i < r; ++i)
        for (uint j = 0; j < c; ++j)
            output_img[x * M + y] += filter[i*c + j] * input_img[(x+i-r/2) * M + (j+y-c/2)];
    return;
}
