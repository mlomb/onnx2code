#include <assert.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "model.h"

void* read_file(const char* filename, long* _size) {
    FILE* fp = fopen(filename, "rb");
    assert(fp != NULL);

    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    void* buffer = malloc(size);

    fread(buffer, sizeof(char), size, fp);
    fclose(fp);

    if (_size)
        *_size = size;

    return buffer;
}

int main(int argc, char** argv) {
    long outputs_size;

    const float* inputs = (const float*)read_file("./sample_inputs.bin", NULL);
    const float* weights = (const float*)read_file("./weights.bin", NULL);
    const float* truth_outputs = (const float*)read_file("./sample_outputs.bin", &outputs_size);
    float* outputs = (float*)malloc(outputs_size);

    float total = 0;
    for (int i = 0; i < 10000; i++) {
        inference(weights, inputs, outputs);

        total += outputs[0];
    }
    
    printf("total: %f\n", total);

    return 0;
}
