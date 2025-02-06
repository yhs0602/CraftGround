#include <iomanip>
#include <iostream>

void printHex(const char *data, size_t data_size) {
    for (size_t i = 0; i < data_size; ++i) {
        // Print the hexadecimal representation of the byte
        std::cout << std::hex << std::setw(2) << std::setfill('0')
                  << (static_cast<unsigned int>(data[i]) & 0xFF) << " ";

        // Print a newline every 16 bytes
        if ((i + 1) % 16 == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::dec << std::endl; // Reset the output format
}