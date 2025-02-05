#pragma once
#ifdef HAS_PNG
#include <png.h>
#include <iostream>
#include <vector>

// https://gist.github.com/dobrokot/10486786
typedef unsigned char ui8;
#define ASSERT_EX(cond, error_message)                                         \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << error_message;                                        \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

void WritePngToMemory(
    size_t w, size_t h, const ui8 *dataRGB, std::vector<ui8> &out
);
#endif