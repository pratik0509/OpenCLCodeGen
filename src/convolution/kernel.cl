void kernel filter(global int* input_arr, global int* output_arr, global int* filter, int size) {
    uint s = get_global_id(0);
    output_arr[s] = 0;
    for (uint i = 0; i < size; ++i)
        output_arr[s] += filter[i] * input_arr[s + i];
    return;
}
