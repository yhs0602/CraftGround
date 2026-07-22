#ifdef HAS_PNG
#include <vector>

#include "png_util.h"

static void
PngWriteCallback(png_structp png_ptr, png_bytep data, png_size_t length) {
    std::vector<ui8> *p = (std::vector<ui8> *)png_get_io_ptr(png_ptr);
    p->insert(p->end(), data, data + length);
}

struct TPngDestructor {
    png_struct *p;
    TPngDestructor(png_struct *p) : p(p) {}
    ~TPngDestructor() {
        if (p) {
            png_destroy_write_struct(&p, NULL);
        }
    }
};

void WritePngToMemory(
    size_t w, size_t h, const ui8 *dataRGB, std::vector<ui8> &out
) {
    out.clear();
    png_structp p =
        png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    ASSERT_EX(p, "png_create_write_struct() failed");
    TPngDestructor destroyPng(p);
    png_infop info_ptr = png_create_info_struct(p);
    ASSERT_EX(info_ptr, "png_create_info_struct() failed");
    ASSERT_EX(0 == setjmp(png_jmpbuf(p)), "setjmp(png_jmpbuf(p) failed");
    png_set_IHDR(
        p,
        info_ptr,
        w,
        h,
        8,
        PNG_COLOR_TYPE_RGB,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );
    // png_set_compression_level(p, 1);
    std::vector<ui8 *> rows(h);
    for (size_t y = 0; y < h; ++y)
        rows[y] = (ui8 *)dataRGB + y * w * 3;
    png_set_rows(p, info_ptr, &rows[0]);
    png_set_write_fn(p, &out, PngWriteCallback, NULL);
    png_write_png(p, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
}
#endif