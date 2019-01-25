__kernel
void matrix_trace(__global float* ARR, __global float S, __global int N, __global int B) {
    float temp;
    int n_index = get_global_id(0);
    int t = N;
    int i;
    while (t > 0) {
        temp = 0;
        for (i = n_index * B; i < N; ++i) {
            temp += ARR[i];
        }
        ARR[n_index/off] = temp;
        t /= B;
        barrier(CLK_GLOBAL_MEM_FENCE);
        if (n_index % B)
            break;
    }
    return;
}