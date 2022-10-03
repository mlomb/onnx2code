#include <unistd.h>
#include <cstdio>

int main(int argc, char **argv) {
    float *weights;

    while(1) {
        // wait for data
        char ready;
        read(STDIN_FILENO, &ready, 1);

        printf("asdasdad\n");

        // mark as ready
        write(STDOUT_FILENO, &ready, 1);
    }

    return 0;
}
