void kernel filter(global int* input, global int* output){
    uint x = get_global_id(0);
    output[x] = input[x];
}
