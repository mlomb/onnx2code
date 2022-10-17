#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <math.h>

#include "model.h"

void* read_file(const char* filename, long* _size) {
    FILE *fp = fopen(filename, "rb");
    assert(fp != NULL);

    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    void* buffer = malloc(size);

    fread(buffer, sizeof(char), size, fp);
    fclose(fp);

    if (_size) *_size = size;

    return buffer;
}

int main(int argc, char **argv) {
    long outputs_size;

    const float* inputs = (const float*) read_file("./sample_inputs.bin", NULL);
    const float* truth_outputs = (const float*) read_file("./sample_outputs.bin", &outputs_size);
    const float* weights = (const float*) read_file("./weights.bin", NULL);
    float* outputs = (float*) malloc(outputs_size);

    inference(weights, inputs, outputs);

    // test against truth
    for (int i = 0; i < outputs_size / 4; i++) {
        if (fabs((double)outputs[i] - (double)truth_outputs[i]) > 1e-5) {
            printf("output[%d] = %f (expected %f)\n", i, outputs[i], truth_outputs[i]);
        }
    }

    return 0;
}
