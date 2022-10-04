#include <unistd.h>
#include <stdio.h>
#include <assert.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

extern void inference(const float* weights, const float* inputs, float* outputs);

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

int main(int argc, char **argv) {
    float *inputs = (float*) map_shared_memory("/onnx2code-inputs");
    float *outputs = (float*) map_shared_memory("/onnx2code-outputs");

    while(1) {
        // wait for data
        char signal;
        read(STDIN_FILENO, &signal, 1);

        // run inference
        inference(NULL, inputs, outputs);

        // mark as ready
        write(STDOUT_FILENO, &signal, 1);
        fsync(STDOUT_FILENO);
    }

    return 0;
}
