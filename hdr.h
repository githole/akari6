#ifndef _HDR_H_
#define _HDR_H_

#include "vec3.h"

#include <algorithm>
#include <vector>

namespace hhdr
{
    using namespace hmath;

    namespace detail
    {
        struct HDRPixel
        {
            unsigned char r, g, b, e;
            HDRPixel(const unsigned char r = 0, const unsigned char g = 0, const unsigned char b = 0, const unsigned char e = 0) :
                r(r), g(g), b(b), e(e) {};

            unsigned char get(int idx) const
            {
                switch (idx) {
                case 0: return r;
                case 1: return g;
                case 2: return b;
                case 3: return e;
                } return 0;
            }

            HDRPixel(const Float3& color)
            {
                double d = std::max(color[0], std::max(color[1], color[2]));
                if (d <= 1e-32) {
                    r = g = b = e = 0;
                    return;
                }
                int ie;
                const double m = frexp(d, &ie); // d = m * 2^e
                d = m * 256.0 / d;

                r = (unsigned char)(color[0] * d);
                g = (unsigned char)(color[1] * d);
                b = (unsigned char)(color[2] * d);
                e = (unsigned char)(ie + 128);
            }
        };
    }

    static void save(const char *filename, float *image, int width, int height, bool absolute = false)
    {
        std::shared_ptr<FILE> fp(fopen(filename, "wb"), [](FILE *f) { if (f != NULL) fclose(f); });

        if (fp == NULL) {
            std::cerr << "Error: " << filename << std::endl;
            return;
        }
        // .hdrフォーマットに従ってデータを書きだす
        // ヘッダ
        unsigned char ret = 0x0a;
        fprintf(fp.get(), "#?RADIANCE%c", (unsigned char)ret);
        // fprintf(fp.get(), "# Made with 100%% pure HDR Shop%c", ret);
        fprintf(fp.get(), "FORMAT=32-bit_rle_rgbe%c", ret);
        fprintf(fp.get(), "EXPOSURE=1.0000000000000%c%c", ret, ret);

        // 輝度値書き出し

        if (width < 8) {
            std::cerr << "Error (Width must be larger than 7): " << filename << std::endl;
            return;
        }

        fprintf(fp.get(), "-Y %d +X %d%c", height, width, ret);

        std::vector<unsigned char> buffer;

        for (int i = 0; i < height; ++i) {
            std::vector<detail::HDRPixel> line;
            for (int j = 0; j < width; ++j) {
                auto index = j + i * width;
                Float3 color(&image[index * 3]);
                if (absolute) {
                    color[0] = abs(color[0]);
                    color[1] = abs(color[1]);
                    color[2] = abs(color[2]);
                }
                detail::HDRPixel p(color);
                line.push_back(p);
            }
            buffer.push_back(0x02);
            buffer.push_back(0x02);
            buffer.push_back((width >> 8) & 0xFF);
            buffer.push_back(width & 0xFF);
            for (int i = 0; i < 4; i++) {
                for (int cursor = 0; cursor < width;) {
                    const int cursor_move = std::min((unsigned int)127, (unsigned int)(width - cursor));
                    buffer.push_back(cursor_move);
                    for (int j = cursor; j < cursor + cursor_move; j++) {
                        buffer.push_back(line[j].get(i));
                    }
                    cursor += cursor_move;
                }
            }
        }
        fwrite(&buffer[0], sizeof(unsigned char), buffer.size(), fp.get());
    }

}

#endif