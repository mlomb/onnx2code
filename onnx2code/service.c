#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "model.h"

void* map_shared_memory(const char* name) {
    int fd = shm_open(name, O_RDWR | O_CREAT, 0666);
    assert(fd != -1);

    // we query the size so we don't have to know it beforehand
    struct stat finfo;
    fstat(fd, &finfo);
    size_t size = finfo.st_size;
    assert(size > 0);

    void* shared = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    assert(shared != MAP_FAILED);
    
    return shared;
}

void* read_file(const char* filename) {
    FILE *fp = fopen(filename, "rb");
    assert(fp != NULL);

    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    void* buffer = malloc(size);

    fread(buffer, sizeof(char), size, fp);
    fclose(fp);

    return buffer;
}

int main(int argc, char **argv) {
    const float* weights = (const float*) read_file(argv[1]);
    float *inputs = (float*) map_shared_memory("/o2c-inputs");
    float *outputs = (float*) map_shared_memory("/o2c-outputs");

    while(1) {
        // wait for data
        char signal;
        read(STDIN_FILENO, &signal, 1);

        // run inference
        inference(weights, inputs, outputs);

        // mark as ready
        write(STDOUT_FILENO, &signal, 1);
        fsync(STDOUT_FILENO);
    }

    return 0;
}
