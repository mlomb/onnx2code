#include <unistd.h>
#include <cstdio>

int main(int argc, char **argv) {
    float *weights;
    
    while(1) {
        // wait for data
        char ready;
        read(STDIN_FILENO, &ready, 1);

        fprintf(stderr,"asdasdad\n");
        sleep(1);

        // mark as ready
        write(STDOUT_FILENO, &ready, 1);
        fsync(STDOUT_FILENO);
    }

    return 0;
}
