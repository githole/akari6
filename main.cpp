#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#include "external/stb/stb_image_write.h"
#include "external/fmath/fmath.hpp"

#include "vec3.h"
#include "util.h"
#include "etc.h"
#include "hdr.h"
#include "test.h"
#include "XYZ.h"
#include "bbox.h"

#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <atomic>

// #define TEST

class FloatImage
{
private:
    size_t width_, height_;
    std::vector<hmath::Double3> data_;
    std::vector<hmath::Double3> data2_;
    std::vector<uint32_t> sample_map_;
public:
    FloatImage(size_t w, size_t h) : width_(w), height_(h), data_(width_ * height_), data2_(width_ * height_), sample_map_(width_ * height_)
    {
    }

    size_t width() const { return width_; }
    size_t height() const { return height_; }

    hmath::Double3& operator()(int x, int y)
    {
        return data_[x + y * width_];
    }

    hmath::Double3& data2(int x, int y)
    {
        return data2_[x + y * width_];
    }

    uint32_t& samples(int x, int y)
    {
        return sample_map_[x + y * width_];
    }
};

namespace
{
    const hmath::Double3 rgb2y(0.2126, 0.7152, 0.0722);
}

// この辺過去のコードもってきただけ
namespace filter
{
    void rgb2YCoCg(float r, float g, float b, float *Y, float *Co, float *Cg) {
        //*Y = 1 / 4.0f * r + 1 / 2.0f * g + 1 / 4.0f * b;
        //*Co = 1 / 2.0f * r + 0 / 1.0f * g - 1 / 2.0f * b;
        //*Cg = -1 / 4.0f * r + 1 / 2.0f * g - 1 / 4.0f * b;
        *Y = r;
        *Co = g;
        *Cg = b;
    }
    void YCoCg2rgb(float Y, float Co, float Cg, float *r, float *g, float *b) {
        /*
        *r = Y + Co - Cg;
        *g = Y + 0 + Cg;
        *b = Y - Co - Cg;
        */
        *r = Y;
        *g = Co;
        *b = Cg;
    }

    void set_vector(std::vector<float>& arr, const int x, const int y, const int ch, const int width, const int height, float *vector, int learn_radius) {
        const int size = learn_radius * 2 + 1;

        for (int oy = -learn_radius; oy <= learn_radius; ++oy) {
            for (int ox = -learn_radius; ox <= learn_radius; ++ox) {
                const int nx = ox + x;
                const int ny = oy + y;
                if (0 <= nx && nx < width && 0 <= ny && ny < height) {
                    vector[(oy + learn_radius) * size + (ox + learn_radius)] = arr[(ny * width + nx) * 3 + ch];
                }
            }
        }
    }

    float length(float *v0, float *v1, int size) {
        float sum = 0;
        for (int i = 0; i < size; ++i) {
            const float a = v0[i] - v1[i];
            sum += a * a;
        }
        return sum;
    }

    void nlm(std::vector<float>& arr0, std::vector<float>& arr1, const int width, const int height) {
        const float sigma = 0.3f;

        // 変換
        for (int iy = 0; iy < height; ++iy) {
            for (int ix = 0; ix < width; ++ix) {
                const int idx = iy * width + ix;
                const float r = arr0[idx * 3 + 0];
                const float g = arr0[idx * 3 + 1];
                const float b = arr0[idx * 3 + 2];
                rgb2YCoCg(r, g, b, &arr0[idx * 3 + 0], &arr0[idx * 3 + 1], &arr0[idx * 3 + 2]);
            }
        }

        // NLM
        for (int iy = 0; iy < height; ++iy) {
            std::cout << "Y: " << iy << "    \r";

            std::vector<std::thread> thread_list;
            int thread_id = 0;
            for (int begin_x = 0; begin_x < width; begin_x += 16) {

                int end_x = begin_x + 16;
                if (end_x >= width)
                    end_x = width;

                std::thread thread([thread_id, begin_x, end_x, iy, width, height, sigma, &arr0, &arr1]() {
                    hetc::set_thread_group(thread_id);

                    const int learn_radius = 3;
                    const int size = learn_radius * 2 + 1;
                    const int compare_raidus = 6;
                    for (int ix = begin_x; ix < end_x; ++ix) {

                        for (int ch = 0; ch < 3; ++ch) {
                            float vector0[size * size] = { 0 };
                            set_vector(arr0, ix, iy, ch, width, height, vector0, learn_radius);

                            const int compare_size = compare_raidus * 2 + 1;
                            float weight_map[compare_size * compare_size] = { 0 };
                            float value_map[compare_size * compare_size] = { 0 };

                            // 探索
                            for (int oy = -compare_raidus; oy <= compare_raidus; ++oy) {
                                for (int ox = -compare_raidus; ox <= compare_raidus; ++ox) {
                                    const int nx = ox + ix;
                                    const int ny = oy + iy;
                                    const int compare_idx = (oy + compare_raidus) * compare_size + (ox + compare_raidus);
                                    if (0 <= nx && nx < width && 0 <= ny && ny < height) {
                                        float vector1[size * size] = { 0 };
                                        set_vector(arr0, nx, ny, ch, width, height, vector1, learn_radius);

                                        // 重み計算
                                        value_map[compare_idx] = arr0[(ny * width + nx) * 3 + ch];
                                        weight_map[compare_idx] = length(vector0, vector1, size * size);
                                    }
                                    else {
                                        weight_map[compare_idx] = -1;
                                    }
                                }
                            }

                            // 結果計算
                            float sum = 0;
                            float total_weight = 0;
                            for (int cy = 0; cy < compare_size; ++cy) {
                                for (int cx = 0; cx < compare_size; ++cx) {
                                    const int compare_idx = cy * compare_size + cx;
                                    if (weight_map[compare_idx] < 0)
                                        continue;
                                    const float weight = exp(-weight_map[compare_idx] / (sigma * sigma));
                                    sum += value_map[compare_idx] * weight;
                                    total_weight += weight;
                                }
                            }
                            if (total_weight > 0)
                                sum /= total_weight;
                            arr1[(iy * width + ix) * 3 + ch] = sum;
                        }
                        const int idx = iy * width + ix;
                        const float Y = arr1[idx * 3 + 0];
                        const float Co = arr1[idx * 3 + 1];
                        const float Cg = arr1[idx * 3 + 2];
                        YCoCg2rgb(Y, Co, Cg, &arr1[idx * 3 + 0], &arr1[idx * 3 + 1], &arr1[idx * 3 + 2]);
                    }
                });
                thread_list.push_back(std::move(thread));
                ++thread_id;
            }

            for (auto& th : thread_list) {
                th.join();
            }
        }
    }

    void median(std::vector<float>& input, std::vector<float>& output, const int width, const int height) 
    {
        for (int iy = 0; iy < height; ++iy) {
            std::cout << "Y: " << iy << "    \r";
            for (int ix = 0; ix < width; ++ix) {
                for (int ch = 0; ch < 3; ++ch) {
                    float p[9] = {};
                    int index = 0;
                    for (int oy = -1; oy <= 1; ++oy) {
                        for (int ox = -1; ox <= 1; ++ox) {

                            int current_x = ix + ox;
                            if (current_x < 0)
                                current_x = 0;
                            else if (width <= current_x)
                                current_x = width - 1;

                            int current_y = iy + oy;
                            if (current_y < 0)
                                current_y = 0;
                            else if (height <= current_y)
                                current_y = height - 1;

                            p[index] = input[(current_x + current_y * width) * 3 + ch];
                            ++index;
                        }
                    }
                    std::sort(p, p + 9);
                    output[(ix + iy * width) * 3 + ch] = p[4];
                }
            }
        }
    }
}

void save_image(const char* filename, FloatImage& image, bool enable_filter = false)
{
    const auto Width = image.width();
    const auto Height = image.height();
    uint64_t total = 0;
    std::vector<float> tonemapped_image(Width * Height * 3);
    std::vector<uint8_t> uint8_tonemapped_image(Width * Height * 3);

    std::vector<float> fdata(Width * Height * 3);
    std::vector<float> fdata2(Width * Height * 3);

    fmath::PowGenerator degamma(1.2f /* ちょいコントラスト強める */ / 2.2f);

    for (int iy = 0; iy < Height; ++iy) {
        for (int ix = 0; ix < Width; ++ix) {
            auto index = ix + iy * Width;
            const double N = image.samples(ix, iy);
            auto p = image(ix, iy) / N;
            total += image.samples(ix, iy);

            //const double lumi = dot(p, rgb2y);
            //p[0] = p[1] = p[2] = image.data2(ix, iy)[0] / (N - 1) - (N / (N - 1)) * lumi * lumi;

            fdata[index * 3 + 0] = (float)p[0];
            fdata[index * 3 + 1] = (float)p[1];
            fdata[index * 3 + 2] = (float)p[2];
        }
    }


    for (int iy = 0; iy < Height; ++iy) {
        for (int ix = 0; ix < Width; ++ix) {
            auto index = ix + iy * Width;
            fdata2[index * 3 + 0] = (degamma.get(hmath::clamp((float)fdata[index * 3 + 0], 0.0f, 1.0f)));
            fdata2[index * 3 + 1] = (degamma.get(hmath::clamp((float)fdata[index * 3 + 1], 0.0f, 1.0f)));
            fdata2[index * 3 + 2] = (degamma.get(hmath::clamp((float)fdata[index * 3 + 2], 0.0f, 1.0f)));
        }
    }
    
    if (enable_filter) {
        // filter::median(fdata2, tonemapped_image, Width, Height);
        for (int iy = 0; iy < Height; ++iy) {
            for (int ix = 0; ix < Width; ++ix) {
                auto index = ix + iy * Width;
                uint8_tonemapped_image[index * 3 + 0] = (uint8_t)(fdata2[index * 3 + 0] * 255);
                uint8_tonemapped_image[index * 3 + 1] = (uint8_t)(fdata2[index * 3 + 1] * 255);
                uint8_tonemapped_image[index * 3 + 2] = (uint8_t)(fdata2[index * 3 + 2] * 255);
            }
        }
    }
    else {
        for (int iy = 0; iy < Height; ++iy) {
            for (int ix = 0; ix < Width; ++ix) {
                auto index = ix + iy * Width;
                uint8_tonemapped_image[index * 3 + 0] = (uint8_t)(fdata2[index * 3 + 0] * 255);
                uint8_tonemapped_image[index * 3 + 1] = (uint8_t)(fdata2[index * 3 + 1] * 255);
                uint8_tonemapped_image[index * 3 + 2] = (uint8_t)(fdata2[index * 3 + 2] * 255);
            }
        }
    }


    double average = (double)total / (Width * Height);
    std::cout << "Average: " << average << " samples/pixel" << std::endl;
    stbi_write_png(filename, (int)Width, (int)Height, 3, uint8_tonemapped_image.data(), (int)(Width * sizeof(uint8_t) * 3));

    // hhdr::save("hoge.hdr", fdata.data(), (int)Width, (int)Height, false);
}

namespace volume
{
    template<typename real>
    constexpr real phase_Henyey_Greenstein(real cost, real g)
    {
        return (float)((1.0 / (4.0 * hmath::pi<real>())) * (1 - g * g) / pow(1 + g * g - 2 * g * cost, 3.0 / 2.0));
    }

    template<typename real, typename rng>
    void sample_phase_HenyayGreenstein(real g, rng& rng, real& theta, real& phi)
    {
        const real u0 = rng.next01();
        const real s = 2.0f * u0 - 1;
        const real k = (1 - g * g) / (1 + g * s);
        const real mu = (1.0f / 2.0f / g) * (1 + g * g - k * k);
        theta = acos(hmath::clamp(mu, (real)-1, (real)1));
        const real u1 = rng.next01();
        phi = 2 * hmath::pi<real>() * u1;
    }

    template<typename real>
    constexpr real phase_Mie(real cost, real g)
    {
        return (1.0f / (4.0f * hmath::pi<real>())) * (3.0f / 2.0f) * (1 - g * g) / (2 + g * g) * (1 + cost * cost) / pow(1 + g * g - 2 * g * cost, 3.0f / 2.0f);
    }

    template<typename real, typename transmittance, typename rng>
    real delta_tracking(const transmittance& coeff, rng& rng)
    {
        const real bound = coeff.bound();
        auto inv_majorant = 1.0f / coeff.majorant();

        real t = 0;
        do {
            t -= /*log(1.0f - rng.next01())*/ fmath::log(1.0f - rng.next01()) * inv_majorant;
        } while (coeff(t) * inv_majorant < rng.next01() && t < bound);

        return t;
    }

    template<typename real, typename transmittance, typename rng>
    real ratio_tracking_estimator(const transmittance& coeff, rng& rng)
    {
        const real bound = coeff.bound();
        auto inv_majorant = 1.0f / coeff.majorant();

        real t = 0;
        real T = 1;
        do {
            t -= /*log(1.0f - rng.next01())*/ fmath::log(1.0f - rng.next01()) * inv_majorant;
            if (t >= bound)
                break;
            T = T * (1.0f - coeff(t) * inv_majorant);
        } while (true);

        return T;
    }


    constexpr float MieScale = 1;
    constexpr float scattering_Mie_base = 2.5f; /* [1/m] */
    constexpr float H_Mie = 10; /* [m] */
    constexpr float Albedo = 0.999f;


    struct HeightExpTable
    {
        static constexpr int N = 256;
        float data[N];

        HeightExpTable()
        {
            for (int i = 0; i < N; ++i) {
                data[i] = exp(-(i * 0.25f) / H_Mie);
            }
        }

        float sample(float h) const
        {
            int i = h * (1.0f / 0.25f);
            if (i < 0)
                i = 0;
            if (N <= i)
                i = N - 1;
            return data[i];
        }
    } heightExpTable;

    struct VolumeTable
    {
        float majorant_ = 0;
        // octree作るため、2ベキかつ全部同じサイズという前提
        static constexpr int NX = 64;
        static constexpr int NY = 64;
        static constexpr int NZ = 64;

        // extinction coeff
        float data[NX * NY * NZ];

        hmath::Float3 index2pos(int x, int y, int z) const
        {
            return hmath::Float3(x - NX / 2.0f, y, z - NZ / 2.0f) + hmath::Float3(1, 1, 1) * 0.5f;
        }

        int pos2index(const hmath::Float3& p) const
        {
            int ix = hmath::clamp((int)(p[0] + NX / 2.0f), 0, NX - 1);
            int iy = hmath::clamp((int)p[1], 0, NY - 1);
            int iz = hmath::clamp((int)(p[2] + NZ / 2.0f), 0, NZ - 1);

            return index2flat(ix, iy, iz);
        }

        int index2flat(int x, int y, int z) const
        {
            return x + y * NX + z * NX * NY;
        }

        // Octreeも作る
        struct Octree
        {
            hrt::BBox box;
            float majorant = 0;
            bool leaf = false;
            Octree* child[2][2][2] = {};
        };

        hmath::Float3 index2pos_bbox(int x, int y, int z) const
        {
            return hmath::Float3(x - NX / 2.0f, y, z - NZ / 2.0f);
        }
        Octree* construct(int depth, int begin_x, int end_x, int begin_y, int end_y, int begin_z, int end_z)
        {
            Octree* current = new Octree; // プログラム終了時まで開放しないのでdeleteしない（行儀悪いね）
            current->box = hrt::BBox(index2pos_bbox(begin_x, begin_y, begin_z), index2pos_bbox(end_x, end_y, end_z));

            int edge_x = (end_x - begin_x) / 2;
            int edge_y = (end_y - begin_y) / 2;
            int edge_z = (end_z - begin_z) / 2;

            if (edge_x == 0 && edge_y == 0 && edge_z == 0) {
                // leaf
                current->leaf = true;
                current->majorant = data[index2flat(begin_x, begin_y, begin_z)];
                return current;
            }

            float max_majorant = -1;
            float min_majorant = std::numeric_limits<float>::infinity();

            for (int iz = 0; iz < 2; ++iz) {
                for (int iy = 0; iy < 2; ++iy) {
                    for (int ix = 0; ix < 2; ++ix) {
                        Octree* child = construct(
                            depth + 1,
                            begin_x + ix * edge_x, begin_x + ix * edge_x + edge_x,
                            begin_y + iy * edge_y, begin_y + iy * edge_y + edge_y,
                            begin_z + iz * edge_z, begin_z + iz * edge_z + edge_z);

                        if (child != nullptr) {
                            current->majorant = std::max(current->majorant, child->majorant);
                            current->child[ix][iy][iz] = child;

                            max_majorant = std::max(max_majorant, child->majorant);
                            min_majorant = std::min(min_majorant, child->majorant);
                        }
                    }
                }
            }

            // もし、childのmajorantの差が一定の幅以下なら、まとめてしまう
            //if ((max_majorant - min_majorant) < 1.0f) {
            if (depth >= 2) {
                current->leaf = true;
                for (int iz = 0; iz < 2; ++iz) {
                    for (int iy = 0; iy < 2; ++iy) {
                        for (int ix = 0; ix < 2; ++ix) {
                            delete current->child[ix][iy][iz];
                            current->child[ix][iy][iz] = nullptr;
                        }
                    }
                }
            }

            return current;
        }

        Octree* root = nullptr;
        void build()
        {
            root = construct(0, 0, NX, 0, NY, 0, NZ);
            // std::cout << "HOGE";
        }

        const Octree* octree_lut[NX * NY * NZ] = {};

        VolumeTable()
        {
            for (int iz = 0; iz < NZ; ++iz) {
                for (int iy = 0; iy < NY; ++iy) {
                    for (int ix = 0; ix < NX; ++ix) {
                        auto p = index2pos(ix, iy, iz);

                        float scale = 1;

                        scale = 0.01f;
                        const float h = p[1];
                        constexpr float solid = 1.0f;

                        if (abs(p[0]) < 3.0f && abs(p[1]) < 2.0f && p[1] > 1.0f)
                            scale = solid;
                        auto k = p - hmath::Float3(-8, 8, -8);
                        if (abs(k[2]) < 3.0f && abs(k[1]) < 5.0f)
                            scale = solid;

                        if (abs(k[2]) < 3.0f && abs(k[1]) < 1.0f)
                            scale = 0.01f;

                        k = p - hmath::Float3(0, 9, 0);
                        if (abs(k[0]) < 4.0f && abs(k[1]) < 3.0f && k[1] > 1.0f)
                            scale = solid;

                        k = p - hmath::Float3(-1, 0, 8);
                        if (abs(k[0]) < 2.0f && abs(k[2]) < 2.0f)
                            scale = solid;

                        const auto extinction_coeff = scale * MieScale * scattering_Mie_base * exp(-h / H_Mie);

                        data[index2flat(ix, iy, iz)] = extinction_coeff;

                        majorant_ = std::max(majorant_, extinction_coeff);
                    }
                }
            }

            build();


            for (int iz = 0; iz < NZ; ++iz) {
                for (int iy = 0; iy < NY; ++iy) {
                    for (int ix = 0; ix < NX; ++ix) {
                        auto p = index2pos(ix, iy, iz);
                        auto ptr = inside(p);
                        octree_lut[index2flat(ix, iy, iz)] = ptr;
                    }
                }
            }
        }

        float majorant() const { return majorant_; }


        float sample(const hmath::Float3& p) const
        {
            if (!root->box.inside(p))
                return 0;

            return data[pos2index(p)];
        }

        const Octree* inside_sub(const Octree* current, const hmath::Float3& p) const
        {
            if (!current->box.inside(p)) {
                return nullptr;
            }

            if (current->leaf) {
                return current;
            }

            //return current;

            for (int iz = 0; iz < 2; ++iz) {
                for (int iy = 0; iy < 2; ++iy) {
                    for (int ix = 0; ix < 2; ++ix) {
                        auto* ptr = current->child[ix][iy][iz];
                        if (ptr) {
                            auto* ret = inside_sub(ptr, p);
                            if (ret)
                                return ret;
                        }
                    }
                }
            }

            return current;
        }

        const Octree* inside(const hmath::Float3& p) const
        {
            return inside_sub(root, p);
        }

        const Octree* fast_inside(const hmath::Float3& p) const
        {
            if (!root->box.inside(p))
                return nullptr;

            return octree_lut[pos2index(p)];
        }
    } volumeTable;

#if 0
    const real bound = coeff.bound();
    auto inv_majorant = 1.0f / coeff.majorant();

    real t = 0;
    do {
        t -= /*log(1.0f - rng.next01())*/ fmath::log(1.0f - rng.next01()) * inv_majorant;
    } while (coeff(t) * inv_majorant < rng.next01() && t < bound);

    return t;
#endif

    float adaptive_delta_tracking(const hmath::Float3& org, const hmath::Float3& dir, float bound, hmath::Rng& rng)
    {
        auto current_org = org;
        auto current_dir = dir;

#if 0
        // delta tracking
        auto inv_majorant = 1 / volumeTable.majorant();
        float t = 0;
        do {
            t -= fmath::log(1.0f - rng.next01()) * inv_majorant;

            if (t > bound)
                return t;

            if (volumeTable.sample(current_org + t * current_dir) * inv_majorant >= rng.next01()) {
                return t;
            }
        } while (1);

#endif

#if 1
        // 初回チェック
        auto* node = volumeTable.fast_inside(current_org);
        if (node == nullptr) {
            float t0, t1;
            if (!volumeTable.root->box.check_intersect(current_org, current_dir, &t0, &t1)) {
                return std::numeric_limits<float>::infinity();
            }

            // volume内部に侵入
            float progress = t0 + 1e-3f;
            current_org = current_org + progress * current_dir;
            return progress + adaptive_delta_tracking(current_org, current_dir, bound - progress, rng);
        }

        float total_t = 0;
        for (;;) {
            auto* node = volumeTable.fast_inside(current_org);
            if (!node)
                return std::numeric_limits<float>::infinity();

            float t0, t1;
            node->box.check_intersect(current_org, current_dir, &t0, &t1);
            if (t0 <= 0.0f)
                t0 = t1;

            // delta tracking
            if (node->majorant == 0) {
                total_t += t0;
                current_org = current_org + (t0 + 1e-3f) * current_dir;
            }
            else {
                auto inv_majorant = 1.0f / node->majorant;
                float t = 0;
                do {
                    t -= fmath::log(1.0f - rng.next01()) * inv_majorant;

                    if (t > t0) {
                        total_t += t0;
                        current_org = current_org + (t0 + 1e-3f) * current_dir;
                        break;
                    }

                    if (total_t + t > bound) {
                        return total_t + t;
                    }

                    if (volumeTable.sample(current_org + t * current_dir) * inv_majorant >= rng.next01()) {
                        return total_t + t;
                    }
                } while (1);
            }
        }
#endif
    }

    template<typename Vec3>
    struct TransmittanceCoeffEvaluator
    {
    private:
        float majorant_;

        static constexpr float a = 0.3f;
        const float cosa = cos(a);
        const float sina = sin(a);

        template<bool majorant = false>
        float extinction_Mie(Vec3 p) const
        {
            float ext_coeff = volumeTable.sample(p);

            if (majorant)
                ext_coeff = volumeTable.majorant();
  
            return ext_coeff;
        }

        /*
        static constexpr float absorp = 0.0001f;

        template<bool majorant = false>
        float absorption_Mie(Vec3 p) const
        {
            return scattering_Mie<majorant>(p) * absorp;
        }
        */
    public:
        Vec3 org;
        Vec3 dir;
        float bnd;

        TransmittanceCoeffEvaluator(const Vec3& o, const Vec3& d, float b) :
            org(o), dir(d), bnd(b)
        {
            //majorant_ = (scattering_Mie<true>({}) + absorption_Mie<true>({}));
            majorant_ = (extinction_Mie<true>({}));
        }

        float majorant() const
        {
            return majorant_;
        }

        float bound() const
        {
            return bnd;
        }

        /*
        float get_scattering_Mie(float t) const
        {
            const auto pos = org + t * dir;
            return scattering_Mie(pos);
        }

        float get_absorption_Mie(float t) const
        {
            const auto pos = org + t * dir;
            return absorption_Mie(pos);
        }
        */

        float operator()(float t) const
        {
            const auto pos = org + t * dir;
            auto sc = extinction_Mie(pos);
            return sc /* + sc * absorp */;
        }
    };

}

namespace integrator
{
    using namespace hmath;

    struct Ray
    {
        Float3 org;
        Float3 dir;
    };

    // from smallpt
    struct Sphere 
    {
        double rad;
        Float3 p;
        Sphere(double rad_, Float3 p_) :
            rad(rad_), p(p_) {}
        double intersect(const Ray &r) const
        { 
            Float3 op = p - r.org;
            double t, eps = 1e-4, b = dot(op, r.dir), det = b * b - dot(op, op) + rad * rad;
            if (det < 0) return 0; else det = sqrt(det);
            return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
        }
    };

    Sphere spheres[] = {
        Sphere(4.0f, Float3(0.0f, 6.0f, 0.0f)),
    };

    bool intersect_plane(const Float3&n, const Float3& o, const Float3& org, const Float3& dir, float &t)
    {
        auto d = dot(n, dir);
        if (d < -1e-6f) {
            t = dot(o - org, n) / d;
            return (t >= 0);
        }

        return false;
    }

    struct IntersectionInfo
    {
        bool hit = false;
        Float3 normal;
        int id = -1;
        float t = std::numeric_limits<float>::infinity();
    };

    IntersectionInfo intersect(const Ray &r)
    {
        //id = 0;
        /*
        double n = sizeof(spheres) / sizeof(Sphere), d, inf = t = 1e20;
        for (int i = int(n);i--;) if ((d = spheres[i].intersect(r)) && d < t) { t = d;id = i; }
        return t < inf;
        */
        //return intersectPlane(Float3(0, 1, 0), Float3(0, 0, 0), r.org, r.dir, t);

        
        IntersectionInfo info;
        float p_t = std::numeric_limits<float>::infinity();
        if (intersect_plane(Float3(0, 1, 0), Float3(0, 0, 0), r.org, r.dir, p_t)) {
            if (p_t < info.t) {
                info.normal = Float3(0, 1, 0);
                info.t = p_t;
                info.id = 0;
            }
        }

        float s_t = (float)spheres[0].intersect(r);
        if (s_t != 0) {
            if (s_t < info.t) {
                info.normal = normalize((r.org + s_t * r.dir) - spheres[0].p);
                info.t = s_t;
                info.id = 1;
            }
        }

        info.hit = info.id != -1;

        return info;
    }

    constexpr float DefaultMinWavelength = 380.0f;
    constexpr float DefaultMaxWavelength = 750.0f;
    constexpr int PlaneSection = 16;
    constexpr float DefaultG = 0.5f;


    struct PlaneTable
    {
        float table[PlaneSection];

        PlaneTable()
        {
            Rng rng;

            for (int i = 0; i < PlaneSection; ++i)
                table[i] = rng.next01() * 4.0f + 8.0f;
        }
    } g_planetable;

    Float3 hsv2rgb(const Float3& hsv)
    {
        auto h = hsv[0];
        auto s = hsv[1];
        auto v = hsv[2];

        float r = v;
        float g = v;
        float b = v;
        if (s > 0.0f) {
            h *= 6.0f;
            int i = (int)h;
            float f = h - (float)i;
            switch (i) {
            default:
            case 0:
                g *= 1 - s * (1 - f);
                b *= 1 - s;
                break;
            case 1:
                r *= 1 - s * f;
                b *= 1 - s;
                break;
            case 2:
                r *= 1 - s;
                b *= 1 - s * (1 - f);
                break;
            case 3:
                r *= 1 - s;
                g *= 1 - s * f;
                break;
            case 4:
                r *= 1 - s * (1 - f);
                g *= 1 - s;
                break;
            case 5:
                g *= 1 - s;
                b *= 1 - s * f;
                break;
            }
        }

        return Float3{ r,g,b };
    }

    Float3 get_plane_emission(float u, float v, float time)
    {
        float length = sqrt(u * u + v * v);

        auto angle = time * 0.005f + (atan2(v, u) + pi<float>()) / (2 * pi<float>());
        const int section = (int)(angle * PlaneSection) % PlaneSection;

        const int length_section = (int)(length / 0.1f);

        if (length < 5.2f) {

            if (4.6f < length && length < 4.9f)
                return 2.0f * Float3(1, 1, 1);

            return {};
        }

        Rng rng;
        rng.set_seed(((uint64_t)section << 32) + length_section);

        if (rng.next() % 32 == 0) {
            return 2.0f * Float3(1, 1, 1);
        }

        if (g_planetable.table[section] < length)
            return {};

        // 色相
        //auto rgb = hsv2rgb(Float3((section + 8) % PlaneSection / (float)PlaneSection, 1, 1));

        //return rgb;

        return 2.0f * Float3(1, 1, 1);
    }

    Float3 cosine_weighted(Rng &random, const Float3& normal, const Float3& tangent, const Float3& binormal) {
        const float phi = random.next01() * 2.0f * pi<float>();
        const float r2 = random.next01(), r2s = sqrt(r2);

        const float tx = r2s * cos(phi);
        const float ty = r2s * sin(phi);
        const float tz = sqrt(1.0f - r2);

        return tz * normal + tx * tangent + ty * binormal;
    }


    // ここに素晴らしいintegratorを書く
    Float3 get_radiance(int w, int h, int x, int y, uint64_t seed)
    {
        Rng rng;
        rng.set_seed(seed);

        const float current_time = rng.next01();

        // data
        constexpr float max_wavelength = DefaultMaxWavelength; /* [nm] */
        constexpr float min_wavelength = DefaultMinWavelength;

        // カメラデータを直接突っ込む
#if 0
        Float3 camera_position(15, 9, 10);
        Float3 camera_dir = normalize(Float3(0, 0, 0) - camera_position);
        const float fovy = (90.0f) / 180.0f * hmath::pi<float>();
#endif
        Float3 camera_position(15, 9, 10);
        Float3 camera_dir = normalize(Float3(3, 0, 1) - camera_position);
        const float fovy = (120.0f) / 180.0f * hmath::pi<float>();

        const float ang = 0.15f;
        Float3 camera_up(sin(ang), cos(ang), 0);

        const auto aspect = (float)w / h;
        const float length = 0.1f;
        const float screen_height = length * tan(fovy / 2);
        const float screen_width = screen_height * aspect;
        const auto screen_side = normalize(cross(camera_dir, camera_up)) * screen_width;
        const auto screen_up = normalize(cross(camera_dir, screen_side)) * screen_height;

        const float U = ((float)(x + rng.next01()) / w) * 2 - 1;
        const float V = ((float)(y + rng.next01()) / h) * 2 - 1;
        const float r = (U * U + V * V);
#if 0
        int ch = rng.next() % 3;
        float table[3] = { -0.05f, 0, 0.05f };
        const float k1 = table[ch];
        const float new_U = U * (1.0f + k1 * r);
        const float new_V = V * (1.0f + k1 * r);
#else
        const float cjitter = rng.next01() * 0.1 - 0.05;
        const float ch = -0.2f + cjitter;

        const float k1 = ch;
        const float new_U = U * (1.0f + k1 * r);
        const float new_V = V * (1.0f + k1 * r);

        const float U2 = U * aspect;
        const float r2 = (U2 * U2 + V * V);
        const float a = 2.0f;
        const float final_weight = a / (a + r2);
#endif

        const auto initial_pos = camera_position + camera_dir * length + new_U * screen_side + new_V * screen_up;
        const auto initial_dir = normalize(initial_pos - camera_position);

#if 1
        // volume integrator!
        static constexpr int MaxDepth = 50;
        Float3 contribution{};
        float weight = 1;
        Ray current_ray{ initial_pos, initial_dir };
        for (int depth = 0; depth < MaxDepth; ++depth) {

            // 開始点 <-> 物体までの距離を求めておく

            auto info = intersect(current_ray);
            if (info.hit) {
                // 何かにヒットした
            }
            else {
                // 背景方向であった
                info.t = 100.0f; // 適当にうちきっちゃる
                // id = -1;
            }

            const float t = info.t;

            // とりあず、物体点からのL * Tについて評価する
            // Tを評価する(1sample track-length estimatorで適当にやる)
            const volume::TransmittanceCoeffEvaluator<Float3> coeff(current_ray.org, current_ray.dir, t);
#if 0
            {
                if (info.id == -1) {
                    // 背景
                }
                else if (info.id == 0) {
                    // surfaceにあたった
                    auto hp = current_ray.org + t * current_ray.dir;
                    auto u = hp[0];
                    auto v = hp[2];

                    auto emission_raw_value = get_plane_emission(u, v, current_time);

                    auto light_distribution = pow(dot(Float3(0, 1, 0), -current_ray.dir), 10.0f) * 10.0f;

                    const float T = volume::ratio_tracking_estimator<float>(coeff, rng);
                    contribution += T * weight * (emission_raw_value + light_distribution * emission_raw_value);
                }
                else {
                    // sphere
                    /*
                    const auto ts = hmath::tangentSpace(info.normal);
                    const Float3 tangent = std::get<0>(ts);
                    const Float3 binormal = std::get<1>(ts);

                    auto next_dir = cosine_weighted(rng, info.normal, tangent, binormal);
                    weight *= 0.99f; // 18%グレー床
                    current_ray = Ray{ current_ray.org + (t - 1e-6f) * current_ray.dir, next_dir };
                    */

                    auto fy = dot(normalize(Float3(0.1f, 0.7f, -0.2f)), (current_ray.org + t * current_ray.dir));
                    const int FY = (int)(fy / 0.1f);

                    Rng rng;
                    rng.set_seed(FY);
                    if (rng.next() % 2) {
                        const float T = volume::ratio_tracking_estimator<float>(coeff, rng);
                        contribution += T * weight * 5 * Float3(1, 1, 1);
                    }
                }

            }

            const auto free_path = volume::delta_tracking<float>(coeff, rng);
            if (free_path > t) {
                break;
            }
            else {
                // 途中で散乱した
                // scattering / absorptionを選ぶ

                const float s_Mie = coeff.get_scattering_Mie(free_path);
                const float a_Mie = coeff.get_absorption_Mie(free_path);

                const float P0 = s_Mie / (s_Mie + a_Mie);
                const float event_p = rng.next01();

                enum Event
                {
                    ScatteringMie,
                    AbsorptionMie,
                } current_event;

                if (event_p < P0) {
                    // Mie Scattering
                    current_event = Event::ScatteringMie;
                }
                else {
                    // Mie absorption
                    current_event = Event::AbsorptionMie;
                    break; // 終了
                }

                // 散乱を続ける

                // phaseに基づいて次の方向を決める

                const auto ts = hmath::tangentSpace(current_ray.dir);

                //                Float3 next_dir;
                const Float3 tangent = std::get<0>(ts);
                const Float3 binormal = std::get<1>(ts);

                float theta, phi;
                volume::sample_phase_HenyayGreenstein(DefaultG, rng, theta, phi);
                const auto next_dir = hmath::polarCoordinateToDirection(theta, phi, current_ray.dir, tangent, binormal);
                weight *= volume::phase_Mie(dot(next_dir, current_ray.dir), DefaultG) / volume::phase_Henyey_Greenstein(cos(theta), DefaultG);

                current_ray = Ray{ current_ray.org + free_path * current_ray.dir, next_dir };
            }
#endif
#if 1
            //auto free_path = volume::delta_tracking<float>(coeff, rng);
            auto free_path = volume::adaptive_delta_tracking(current_ray.org, current_ray.dir, t, rng);

            if (free_path > t) {
                if (info.id == -1) {
                    // 背景
                    break;
                }

                if (info.id == 0) {
                    // surfaceにあたった
                    auto hp = current_ray.org + t * current_ray.dir;
                    auto u = hp[0];
                    auto v = hp[2];

                    auto emission_raw_value = get_plane_emission(u, v, current_time);

                    auto light_distribution = pow(dot(Float3(0, 1, 0), -current_ray.dir), 10.0f) * 10.0f;

                    contribution += weight * (emission_raw_value + light_distribution * emission_raw_value);
                    break;
                }
                else {
                    // sphere
                    /*
                    const auto ts = hmath::tangentSpace(info.normal);
                    const Float3 tangent = std::get<0>(ts);
                    const Float3 binormal = std::get<1>(ts);

                    auto next_dir = cosine_weighted(rng, info.normal, tangent, binormal);
                    weight *= 0.99f; // 18%グレー床
                    current_ray = Ray{ current_ray.org + (t - 1e-6f) * current_ray.dir, next_dir };
                    */

                    auto fy = dot(normalize(Float3(0.1f, 0.7f, -0.2f)), (current_ray.org + t * current_ray.dir));
                    const int FY = (int)(fy / 0.1f);

                    Rng rng;
                    rng.set_seed(FY);
                    if (rng.next() % 2) {
                        contribution += weight * 5 * Float3(1, 1, 1);
                    }

                    break;
                }

#if 0
                // cosine weighted sampling!!
                auto next_dir = cosine_weighted(rng, Float3(0, 1, 0), Float3(1, 0, 0), Float3(0, 0, 1));
                weight *= 0.99f; // 18%グレー床
                current_ray = Ray{ current_ray.org + (t - 1e-6f) * current_ray.dir, next_dir };
#endif
            }
            else {
                // 途中で散乱した
                // scattering / absorptionを選ぶ

                const float P0 = volume::Albedo;
                const float event_p = rng.next01();

                enum Event
                {
                    ScatteringMie,
                    AbsorptionMie,
                } current_event;

                if (event_p < P0) {
                    // Mie Scattering
                    current_event = Event::ScatteringMie;
                }
                else {
                    // Mie absorption
                    current_event = Event::AbsorptionMie;
                    break; // 終了
                }

                // 散乱を続ける

                // phaseに基づいて次の方向を決める

                const auto ts = hmath::tangentSpace(current_ray.dir);

//                Float3 next_dir;
                const Float3 tangent = std::get<0>(ts);
                const Float3 binormal = std::get<1>(ts);

                float theta, phi;
                volume::sample_phase_HenyayGreenstein(DefaultG, rng, theta, phi);
                const auto next_dir = hmath::polarCoordinateToDirection(theta, phi, current_ray.dir, tangent, binormal);
                weight *= volume::phase_Mie(dot(next_dir, current_ray.dir), DefaultG) / volume::phase_Henyey_Greenstein(cos(theta), DefaultG);

                current_ray = Ray{ current_ray.org + free_path * current_ray.dir, next_dir };
            }
#endif
        }

#if 0
        Float3 coeff(0, 0, 0);
        coeff[ch] = 1;
        contribution = product(contribution, coeff);
        return contribution;
#endif
        Float3 coeff(1, 1, 1);
        if (cjitter < 0) {
            coeff = lerp(coeff, Float3(1, 0.4, 0.1), -cjitter * 10);
        }
        else {
            coeff = lerp(coeff, Float3(0.1, 0.6, 1), cjitter * 10);
        }

        contribution = final_weight * product(contribution, coeff);
        return contribution;


#endif

#if 0
        // 交差判定の適当なテスト
        Ray ray{ initial_pos, initial_dir };
        float t;
        int id;
        if (intersect(ray, t, id)) {
            auto hp = ray.org + t * ray.dir;

            auto u = hp[0];
            auto v = hp[2];
            return get_plane_emission(u, v);
        }
        else {
            return {};
        }
#endif
    }
}


#ifdef TEST

int main(int argc, char** argv)
{
    const int Width = 1920;
    const int Height = 1080;

    FloatImage image(Width, Height);

    for (int iy = 0; iy < Height; ++iy) {
        for (int ix = 0; ix < Width; ++ix) {
            auto ret = integrator::get_radiance(Width, Height, ix, iy, ix + iy * Width);

            image(ix, iy) += ret;
#if 0
            auto wl = integrator::get_radiance(Width, Height, ix, iy, ix + iy * Width);


            double xyz_cf[3];
            hcolor::get_CIE_XYZ_1931(wl, xyz_cf);

            /*
            double xyz[3];
            for (int ch = 0; ch < 3; ++ch) {
                xyz[ch] = (contribution / NumEstimationSample) * xyz_cf[ch] / (1.0 / (max_wavelength - min_wavelength));
            }
            */

            double rgb[3];
            hcolor::CIE_XYZ_to_sRGB(xyz_cf, rgb);
            for (int ch = 0; ch < 3; ++ch) {
                image(ix, iy)[ch] = std::max(0.0f, (float)rgb[ch]);
            }
#endif
//            image(ix, iy) = ret;
            image.samples(ix, iy) += 1;
        }
    }

    save_image("test.png", image);

    return 0;
}

#else

static constexpr double bluenoise[64 * 64] = {
    0.849852,0.486470,0.362653,0.135864,0.875363,0.575351,0.301758,0.075770,0.161739,0.800330,0.847244,0.518346,0.404423,0.904120,0.267243,0.146202,0.355750,0.898594,0.678625,0.443323,0.033950,0.932254,0.371548,0.292922,0.908567,0.423293,0.514051,0.793810,0.929755,0.304117,0.511527,0.113687,0.826446,0.466765,0.187289,0.112447,0.394193,0.484879,0.944380,0.730528,0.573809,0.117276,0.494215,0.730491,0.343190,0.544931,0.850984,0.515653,0.224940,0.697636,0.520862,0.900162,0.597412,0.342174,0.971593,0.615412,0.241650,0.714757,0.392243,0.715624,0.188322,0.767780,0.048297,0.283205,
    0.993331,0.214360,0.233942,0.804800,0.614341,0.943177,0.410993,0.910178,0.556641,0.708375,0.201572,0.146765,0.800078,0.653372,0.964413,0.694654,0.447921,0.956558,0.267971,0.064932,0.747236,0.182179,0.782422,0.518113,0.140819,0.998986,0.203433,0.609959,0.050403,0.732710,0.227498,0.045286,0.519645,0.925596,0.697328,0.167991,0.780614,0.017790,0.294826,0.080991,0.846122,0.323868,0.923337,0.959860,0.636513,0.071861,0.940476,0.584835,0.146124,0.653570,0.379293,0.265867,0.077799,0.418154,0.875848,0.138673,0.301351,0.548373,0.850567,0.098059,0.654805,0.884954,0.501749,0.178861,
    0.400652,0.975516,0.846758,0.080618,0.152660,0.833245,0.361023,0.042597,0.398736,0.997767,0.609109,0.334376,0.391058,0.742418,0.504235,0.544300,0.014530,0.607537,0.195390,0.645519,0.421857,0.848747,0.554305,0.045695,0.663247,0.831429,0.099037,0.429314,0.459044,0.872431,0.982293,0.693931,0.311785,0.422100,0.056854,0.652806,0.516034,0.887702,0.559528,0.400066,0.142135,0.268065,0.690012,0.146968,0.479044,0.174231,0.373814,0.309132,0.294940,0.997614,0.795623,0.738729,0.790249,0.063967,0.506912,0.650584,0.927992,0.039786,0.986678,0.006011,0.420657,0.317803,0.271814,0.683366,
    0.046924,0.759705,0.439970,0.510794,0.719580,0.266475,0.542444,0.670824,0.289723,0.471790,0.125453,0.683408,0.097383,0.996944,0.210988,0.116944,0.322126,0.811307,0.474585,0.590914,0.292710,0.089056,0.945044,0.463161,0.197111,0.320706,0.568482,0.733662,0.661809,0.076658,0.223061,0.414840,0.951565,0.229846,0.364500,0.854915,0.546588,0.250469,0.111070,0.660195,0.524278,0.802171,0.044082,0.790829,0.517145,0.157355,0.547256,0.780727,0.874625,0.039419,0.478386,0.125986,0.438911,0.722690,0.281075,0.820747,0.171070,0.395007,0.735624,0.283311,0.555509,0.825919,0.122530,0.607168,
    0.347354,0.225270,0.648627,0.002475,0.994180,0.591356,0.062768,0.848043,0.755032,0.923615,0.188774,0.859986,0.027341,0.359236,0.816749,0.441946,0.788971,0.728392,0.054971,0.161056,0.676361,0.388394,0.323201,0.721071,0.792773,0.387394,0.961814,0.218465,0.830945,0.345456,0.494587,0.840892,0.721730,0.052872,0.293888,0.687077,0.128832,0.389219,0.769214,0.190129,0.996184,0.765063,0.228372,0.873255,0.649914,0.897493,0.219937,0.457668,0.685477,0.161123,0.897081,0.604762,0.951470,0.327066,0.050631,0.678798,0.594874,0.065200,0.691761,0.456656,0.614642,0.942488,0.781339,0.534319,
    0.876661,0.129491,0.547098,0.381533,0.872984,0.222684,0.418094,0.093150,0.182715,0.444474,0.205197,0.626687,0.467381,0.192054,0.929014,0.574243,0.137720,0.398632,0.947779,0.889019,0.541203,0.786426,0.144505,0.076575,0.997338,0.137341,0.231220,0.453969,0.565982,0.007195,0.154364,0.121808,0.544539,0.583001,0.386553,0.808332,0.317624,0.982280,0.859275,0.268611,0.593596,0.175369,0.494979,0.330084,0.084843,0.728841,0.007806,0.618768,0.339234,0.766315,0.459722,0.189937,0.669731,0.868981,0.213732,0.903293,0.372339,0.803609,0.406913,0.892685,0.055953,0.334955,0.139074,0.474411,
    0.974970,0.085827,0.728456,0.748967,0.277999,0.811838,0.935727,0.365446,0.646501,0.546286,0.012487,0.790117,0.895005,0.534837,0.238489,0.703640,0.579725,0.317026,0.109827,0.711382,0.206272,0.910752,0.613907,0.500502,0.009279,0.257624,0.680371,0.908962,0.095106,0.630505,0.502920,0.752455,0.978184,0.677049,0.144487,0.932047,0.102783,0.453136,0.652429,0.501106,0.708435,0.333468,0.947280,0.668522,0.117873,0.509820,0.969438,0.132442,0.944381,0.400455,0.282131,0.025400,0.130798,0.396155,0.524884,0.092278,0.466035,0.053165,0.271348,0.232160,0.842192,0.406560,0.554516,0.666823,
    0.798240,0.452605,0.677332,0.172737,0.502241,0.115548,0.651668,0.473277,0.838385,0.294468,0.959957,0.701960,0.341827,0.086583,0.864939,0.676582,0.214200,0.505755,0.644426,0.723831,0.402608,0.193030,0.709458,0.838555,0.930982,0.777773,0.404761,0.498552,0.795978,0.286007,0.845809,0.190813,0.343127,0.060882,0.466025,0.695407,0.620250,0.163054,0.790566,0.116537,0.003162,0.845832,0.918790,0.384950,0.856214,0.780725,0.334577,0.261747,0.787761,0.694588,0.916349,0.567118,0.749751,0.482550,0.835683,0.298897,0.589660,0.955576,0.687278,0.521267,0.615506,0.175242,0.457346,0.050053,
    0.310903,0.592691,0.356048,0.855940,0.538782,0.013218,0.255187,0.883384,0.158443,0.717841,0.403511,0.260490,0.662439,0.481636,0.117098,0.040310,0.840419,0.346687,0.058264,0.928895,0.118807,0.505861,0.189443,0.431961,0.245508,0.605670,0.191440,0.996546,0.652170,0.318640,0.791146,0.020419,0.562379,0.839434,0.244764,0.555316,0.036828,0.194345,0.947404,0.409734,0.277121,0.688197,0.025226,0.548649,0.054239,0.398278,0.635844,0.449792,0.653193,0.039761,0.205538,0.424270,0.998373,0.226655,0.630545,0.968321,0.048558,0.850787,0.355842,0.090370,0.957076,0.180394,0.745785,0.674492,
    0.197929,0.049841,0.890386,0.301418,0.781262,0.628255,0.397405,0.074851,0.558368,0.213048,0.068115,0.915997,0.208598,0.750806,0.951634,0.420066,0.705296,0.241350,0.957651,0.545680,0.385701,0.873768,0.148532,0.603429,0.015418,0.355496,0.161260,0.720778,0.476416,0.105437,0.591027,0.930575,0.651615,0.385965,0.987218,0.333082,0.933946,0.513373,0.724879,0.293670,0.581223,0.425992,0.188544,0.315814,0.736868,0.102363,0.878112,0.837972,0.140271,0.989481,0.352487,0.559835,0.818038,0.101198,0.709187,0.177465,0.730204,0.264219,0.506662,0.799630,0.038959,0.333059,0.989682,0.436424,
    0.490698,0.963878,0.705242,0.418796,0.099302,0.194189,0.689475,0.757692,0.974443,0.477992,0.669692,0.350297,0.791388,0.605425,0.289234,0.471022,0.915262,0.245939,0.108823,0.629075,0.786381,0.983151,0.702554,0.313449,0.800458,0.748688,0.052456,0.926602,0.342184,0.257564,0.881633,0.534181,0.281464,0.775866,0.142876,0.717803,0.849502,0.608413,0.266321,0.047202,0.841305,0.780260,0.131205,0.654206,0.795103,0.568412,0.163157,0.872371,0.291005,0.471559,0.739513,0.067381,0.672665,0.887440,0.548395,0.467077,0.122601,0.442033,0.136131,0.659846,0.739478,0.866137,0.572601,0.811520,
    0.144073,0.032952,0.270215,0.571943,0.934439,0.903530,0.514061,0.263237,0.417029,0.827198,0.073678,0.901935,0.496904,0.956338,0.022011,0.150685,0.513818,0.770353,0.878477,0.215206,0.002097,0.203441,0.510621,0.176739,0.458474,0.877942,0.422151,0.647371,0.733028,0.526702,0.165578,0.830308,0.063966,0.173772,0.301815,0.472556,0.084126,0.977476,0.910032,0.187030,0.491334,0.948906,0.516516,0.254662,0.906360,0.068353,0.355443,0.534371,0.259210,0.801932,0.161590,0.596859,0.179327,0.272983,0.389343,0.623709,0.008178,0.583181,0.841421,0.915247,0.377476,0.231022,0.060175,0.712917,
    0.616365,0.373210,0.801750,0.234240,0.673339,0.027799,0.566027,0.884341,0.548796,0.297265,0.614751,0.165986,0.438797,0.562888,0.211465,0.949994,0.685072,0.578635,0.055199,0.427753,0.833102,0.334502,0.657421,0.863431,0.049529,0.001237,0.584331,0.278889,0.829244,0.998775,0.368670,0.631451,0.376884,0.959387,0.696930,0.798325,0.203800,0.340556,0.763496,0.413124,0.103077,0.687814,0.366788,0.021016,0.442233,0.746317,0.974181,0.678440,0.014408,0.646888,0.934256,0.335958,0.522936,0.959051,0.320633,0.818556,0.956123,0.285842,0.423957,0.126785,0.450758,0.140675,0.507107,0.935326,
    0.463127,0.985368,0.154962,0.344874,0.503980,0.367554,0.954967,0.169024,0.114041,0.897137,0.224830,0.971644,0.720781,0.303895,0.086655,0.862970,0.377871,0.126956,0.686680,0.148941,0.555607,0.500141,0.767558,0.161786,0.343608,0.550924,0.690102,0.073580,0.469614,0.621540,0.106272,0.235725,0.546171,0.451575,0.890658,0.631420,0.025140,0.504085,0.626136,0.238808,0.585763,0.817554,0.913887,0.765608,0.522029,0.178692,0.274629,0.061617,0.797718,0.481665,0.271994,0.803138,0.695495,0.140835,0.646269,0.225436,0.101983,0.559337,0.708201,0.579011,0.750221,0.829478,0.677616,0.280748,
    0.730172,0.180758,0.557784,0.081821,0.720626,0.629389,0.735705,0.447574,0.798943,0.594868,0.324460,0.470370,0.793026,0.496324,0.636770,0.723445,0.273888,0.971985,0.408734,0.821924,0.243493,0.969278,0.113143,0.578399,0.963274,0.434260,0.942433,0.315035,0.277093,0.945246,0.729702,0.430766,0.932240,0.054921,0.654703,0.347195,0.555720,0.150353,0.827871,0.068844,0.982387,0.297235,0.122549,0.818493,0.210875,0.382521,0.828214,0.901203,0.373696,0.690780,0.089430,0.864423,0.440295,0.777650,0.083455,0.935730,0.467881,0.153752,0.807797,0.006920,0.259550,0.075080,0.400879,0.037001,
    0.864123,0.290293,0.877473,0.687626,0.441898,0.892767,0.192666,0.312087,0.983538,0.688450,0.788463,0.122512,0.061647,0.571892,0.203986,0.429188,0.156585,0.567004,0.619760,0.724374,0.907863,0.297846,0.693223,0.709742,0.029982,0.376156,0.132166,0.782605,0.578607,0.193112,0.009207,0.661657,0.135782,0.838603,0.101194,0.174185,0.944327,0.288490,0.699081,0.905940,0.497300,0.152302,0.442864,0.932284,0.676294,0.954247,0.554763,0.177217,0.605639,0.146801,0.964559,0.531044,0.203454,0.231554,0.405715,0.690916,0.611328,0.888947,0.346029,0.936576,0.464963,0.969603,0.562959,0.659988,
    0.977122,0.538484,0.132590,0.964980,0.075004,0.303809,0.606250,0.104320,0.407967,0.011404,0.296640,0.187244,0.827695,0.907484,0.362945,0.683315,0.850004,0.238434,0.434388,0.129401,0.663266,0.044792,0.569042,0.425708,0.841460,0.752766,0.630179,0.251323,0.838008,0.604552,0.796848,0.380314,0.756972,0.576763,0.293242,0.436176,0.813060,0.717249,0.384272,0.279079,0.554131,0.762602,0.537163,0.000464,0.119619,0.465215,0.238412,0.277975,0.456817,0.816387,0.412571,0.049364,0.605465,0.650790,0.904395,0.815366,0.293802,0.397315,0.072976,0.668526,0.146443,0.878934,0.165156,0.209888,
    0.095949,0.397910,0.577340,0.797375,0.198575,0.045913,0.838518,0.141665,0.548169,0.867997,0.604299,0.962748,0.504419,0.629105,0.814485,0.110900,0.463483,0.975998,0.089483,0.371857,0.867072,0.403325,0.258510,0.920741,0.207511,0.336577,0.521711,0.112523,0.058581,0.920508,0.439353,0.296430,0.962632,0.883244,0.658055,0.004227,0.496886,0.188368,0.625174,0.034307,0.408204,0.882182,0.637128,0.327219,0.185140,0.747352,0.848129,0.026527,0.727123,0.069462,0.240160,0.976477,0.893102,0.294293,0.476506,0.054546,0.736508,0.481357,0.801692,0.239805,0.311815,0.506995,0.665599,0.772387,
    0.464848,0.850412,0.713662,0.317208,0.543626,0.339503,0.648081,0.747491,0.403571,0.130619,0.346877,0.648975,0.231738,0.429177,0.272817,0.022312,0.212332,0.565025,0.893630,0.174691,0.514489,0.743725,0.063790,0.554066,0.660967,0.852705,0.990325,0.772622,0.490360,0.670909,0.246794,0.538548,0.409709,0.091746,0.603886,0.919877,0.345609,0.069672,0.761974,0.091973,0.834204,0.978487,0.248277,0.954149,0.805237,0.372240,0.932531,0.583119,0.967914,0.416627,0.621516,0.690978,0.539200,0.108045,0.662039,0.175835,0.894641,0.191341,0.643243,0.557318,0.256022,0.796647,0.382492,0.926650,
    0.244958,0.034594,0.114437,0.984202,0.229429,0.912780,0.505367,0.910409,0.271062,0.031845,0.540679,0.817389,0.079633,0.895780,0.969362,0.730365,0.665713,0.250217,0.766620,0.018255,0.929125,0.449614,0.681788,0.039779,0.149746,0.399103,0.211612,0.320796,0.162408,0.003500,0.749099,0.891374,0.229127,0.518997,0.182310,0.719971,0.466573,0.638158,0.951643,0.688144,0.522899,0.362196,0.057785,0.560501,0.617109,0.255946,0.414857,0.173870,0.669361,0.317763,0.850483,0.172502,0.735095,0.485245,0.858781,0.569749,0.257777,0.968299,0.858238,0.101218,0.453963,0.962031,0.198036,0.547171,
    0.666990,0.917915,0.400531,0.638131,0.751794,0.033611,0.196166,0.554165,0.783940,0.993494,0.177026,0.717004,0.570074,0.329620,0.501844,0.427785,0.861560,0.341064,0.151785,0.580560,0.316698,0.781531,0.854297,0.272001,0.475910,0.784468,0.689184,0.867331,0.712770,0.339952,0.191092,0.078919,0.788691,0.871077,0.584836,0.906862,0.287840,0.154083,0.254235,0.173835,0.503043,0.737532,0.325529,0.136150,0.877446,0.163811,0.766269,0.042494,0.505663,0.448865,0.066866,0.255963,0.396070,0.337072,0.068178,0.074321,0.658609,0.375286,0.134037,0.914723,0.616238,0.295687,0.059183,0.795789,
    0.005097,0.600852,0.465569,0.328164,0.133337,0.530638,0.375251,0.074516,0.606995,0.458250,0.864305,0.397191,0.208703,0.828976,0.056851,0.795248,0.572102,0.643763,0.511183,0.980795,0.213623,0.108746,0.386582,0.903500,0.605614,0.296000,0.507679,0.216527,0.939928,0.506549,0.487725,0.446753,0.713335,0.273132,0.062646,0.126450,0.969970,0.894713,0.522193,0.748406,0.051662,0.201041,0.876867,0.523192,0.453560,0.905664,0.215521,0.690538,0.253834,0.863655,0.995683,0.666069,0.885070,0.785536,0.450071,0.914005,0.938881,0.617763,0.564060,0.019520,0.755194,0.699301,0.949310,0.356883,
    0.178716,0.861424,0.526014,0.825811,0.874710,0.608073,0.826672,0.998728,0.237870,0.733108,0.331294,0.030942,0.954637,0.660796,0.122545,0.249319,0.182239,0.376601,0.086915,0.796353,0.879953,0.532085,0.646617,0.087070,0.539443,0.991631,0.806719,0.562909,0.043403,0.830643,0.764361,0.233446,0.925597,0.626364,0.999054,0.337322,0.565110,0.821653,0.200735,0.598821,0.432639,0.370736,0.672348,0.591049,0.951377,0.320279,0.532357,0.944815,0.577231,0.385138,0.152764,0.899564,0.588070,0.117191,0.208633,0.815868,0.150010,0.359705,0.829569,0.489209,0.133492,0.214638,0.986664,0.761586,
    0.366045,0.233814,0.700953,0.054136,0.426528,0.756492,0.943391,0.100325,0.658532,0.495112,0.128354,0.603999,0.368031,0.519168,0.333002,0.911941,0.985725,0.863106,0.491926,0.286379,0.010722,0.354214,0.734683,0.496618,0.209452,0.076625,0.651430,0.105724,0.620248,0.321510,0.993883,0.125072,0.352017,0.444126,0.501548,0.788682,0.673494,0.042082,0.391970,0.296256,0.806224,0.927205,0.750644,0.042169,0.904937,0.383912,0.022598,0.727376,0.804907,0.338677,0.045075,0.626850,0.002236,0.514171,0.397293,0.687037,0.506430,0.204484,0.411703,0.911174,0.306226,0.623096,0.429466,0.485186,
    0.886334,0.095994,0.952820,0.186029,0.307851,0.261312,0.541864,0.897048,0.309904,0.266612,0.987808,0.842969,0.719708,0.051787,0.588707,0.733261,0.414693,0.598112,0.064381,0.681520,0.738128,0.998442,0.034480,0.780462,0.433011,0.368206,0.878529,0.778523,0.445453,0.147608,0.379829,0.613995,0.803634,0.021094,0.175327,0.421860,0.141246,0.747259,0.532680,0.983268,0.074456,0.481747,0.173708,0.248220,0.813068,0.563726,0.200709,0.411479,0.093891,0.512255,0.879456,0.447882,0.807109,0.939038,0.654352,0.273561,0.953341,0.798617,0.714046,0.025148,0.688776,0.078264,0.149824,0.832758,
    0.034611,0.668820,0.569461,0.498887,0.886785,0.139580,0.387705,0.644471,0.010257,0.766516,0.057940,0.433731,0.150092,0.992648,0.815851,0.152509,0.684885,0.235636,0.317393,0.833818,0.162807,0.251418,0.854945,0.664557,0.167167,0.662115,0.244116,0.519366,0.682137,0.079548,0.919145,0.786751,0.571796,0.919464,0.639438,0.381813,0.860618,0.655675,0.091186,0.225523,0.707146,0.628945,0.532975,0.595661,0.286282,0.869477,0.988189,0.642383,0.893881,0.738630,0.260694,0.191677,0.358566,0.167429,0.752753,0.014474,0.552038,0.114292,0.374623,0.942296,0.161751,0.879560,0.306271,0.564971,
    0.626194,0.319632,0.992515,0.796847,0.457254,0.071670,0.698691,0.911399,0.514337,0.462053,0.327908,0.925188,0.639873,0.173984,0.262606,0.401083,0.032331,0.894267,0.951188,0.649530,0.522475,0.629000,0.469800,0.937083,0.913724,0.273992,0.019458,0.181810,0.810338,0.378904,0.225938,0.282561,0.242101,0.101588,0.808716,0.713011,0.924852,0.188606,0.269207,0.906194,0.428022,0.008737,0.736212,0.114489,0.915677,0.511959,0.052428,0.253076,0.472158,0.985303,0.537515,0.792643,0.579364,0.034114,0.522515,0.252600,0.876885,0.678202,0.786915,0.260977,0.530379,0.474757,0.948835,0.443466,
    0.229574,0.371340,0.082088,0.718393,0.571004,0.979758,0.262139,0.743894,0.141224,0.711140,0.618204,0.104214,0.465009,0.773019,0.578875,0.978458,0.652173,0.358867,0.461651,0.097796,0.392839,0.319406,0.154165,0.005457,0.542411,0.836925,0.922937,0.768960,0.484001,0.969382,0.575001,0.729384,0.431172,0.756256,0.510329,0.145622,0.439926,0.472957,0.528399,0.765253,0.815925,0.383153,0.218887,0.828627,0.355182,0.678186,0.750309,0.181952,0.691055,0.319948,0.125882,0.449250,0.996706,0.281708,0.921956,0.707513,0.432922,0.201174,0.329483,0.983815,0.593598,0.100916,0.816504,0.747595,
    0.535799,0.965877,0.217742,0.006048,0.294032,0.166661,0.837213,0.422799,0.211582,0.071272,0.876412,0.180108,0.361581,0.111045,0.907162,0.526567,0.127249,0.288056,0.756985,0.883178,0.077857,0.813383,0.769300,0.686928,0.313956,0.417384,0.599299,0.135708,0.262076,0.386538,0.831164,0.002933,0.688259,0.959612,0.005613,0.261873,0.602946,0.340870,0.116932,0.601487,0.143939,0.937727,0.678921,0.970753,0.000541,0.226225,0.497089,0.985691,0.044649,0.463037,0.613169,0.930634,0.784032,0.502540,0.957087,0.047603,0.653847,0.044888,0.520510,0.873670,0.171730,0.729864,0.330490,0.033638,
    0.586473,0.830158,0.422216,0.674189,0.863282,0.546620,0.938672,0.603475,0.273397,0.951991,0.815856,0.734254,0.593617,0.328218,0.709929,0.264810,0.776401,0.564004,0.095907,0.671019,0.254541,0.943702,0.367642,0.085422,0.818928,0.202322,0.461476,0.634453,0.726907,0.544663,0.103721,0.116049,0.222128,0.304430,0.603159,0.725770,0.906118,0.059469,0.861633,0.815274,0.471458,0.314135,0.558678,0.077044,0.346312,0.610763,0.857123,0.373499,0.783949,0.894950,0.250117,0.329624,0.099998,0.197410,0.404994,0.543233,0.084820,0.836520,0.620598,0.964951,0.471057,0.127007,0.637090,0.840869,
    0.386739,0.179431,0.786699,0.497430,0.124252,0.329331,0.722931,0.030428,0.445255,0.334416,0.567199,0.025155,0.495915,0.190671,0.808326,0.426491,0.884369,0.210545,0.973253,0.615014,0.106661,0.571720,0.498126,0.642032,0.980300,0.792695,0.087825,0.251958,0.161588,0.786364,0.370538,0.889706,0.444963,0.935012,0.133478,0.843304,0.333037,0.450716,0.616543,0.231542,0.039392,0.887466,0.690060,0.165816,0.721250,0.503550,0.296863,0.155617,0.633790,0.916848,0.673558,0.804914,0.565727,0.583353,0.308567,0.658217,0.924269,0.284758,0.426856,0.253869,0.372928,0.902196,0.717293,0.201149,
    0.953857,0.100550,0.272422,0.607478,0.987664,0.238885,0.798334,0.295204,0.923565,0.422454,0.881060,0.263816,0.659602,0.920997,0.127839,0.638218,0.028313,0.815287,0.325522,0.169140,0.374020,0.887761,0.322805,0.137387,0.550905,0.358724,0.871387,0.689028,0.935423,0.513208,0.290621,0.634062,0.598625,0.712026,0.826714,0.579079,0.244955,0.999482,0.072644,0.654481,0.750772,0.361305,0.651800,0.992084,0.241700,0.051285,0.812987,0.554306,0.285781,0.125397,0.503321,0.159168,0.873194,0.031379,0.720833,0.981134,0.391991,0.179461,0.769415,0.545736,0.108517,0.573711,0.497701,0.336315,
    0.433601,0.697239,0.031927,0.893202,0.568254,0.417417,0.147266,0.512362,0.059816,0.768541,0.173471,0.978898,0.332142,0.448892,0.330402,0.977536,0.495014,0.451203,0.081825,0.967996,0.665681,0.219746,0.904230,0.015059,0.437830,0.619633,0.129342,0.403308,0.657393,0.064228,0.245518,0.794335,0.209980,0.423524,0.178458,0.329382,0.689261,0.134868,0.442790,0.899410,0.400946,0.010832,0.518618,0.259907,0.781580,0.956172,0.886183,0.693299,0.095467,0.428826,0.971541,0.191859,0.456880,0.808003,0.219039,0.122768,0.486630,0.680942,0.015052,0.705750,0.946345,0.444379,0.037687,0.835106,
    0.922587,0.550901,0.734324,0.327736,0.655406,0.018555,0.803036,0.269905,0.721909,0.542796,0.638463,0.117343,0.037856,0.808090,0.518192,0.221715,0.163993,0.725213,0.562054,0.783652,0.480049,0.931861,0.722866,0.958295,0.243425,0.814977,0.986840,0.498898,0.373649,0.941784,0.743220,0.031287,0.967523,0.875267,0.086396,0.558517,0.468362,0.928378,0.616084,0.526103,0.162295,0.119721,0.810713,0.419569,0.090443,0.679306,0.435050,0.298983,0.868403,0.774188,0.117617,0.555484,0.645536,0.943649,0.501890,0.286970,0.554957,0.856338,0.364528,0.234149,0.830515,0.123074,0.278630,0.751336,
    0.416156,0.150098,0.062641,0.397101,0.931540,0.862810,0.202011,0.992532,0.371334,0.848238,0.014278,0.566806,0.160536,0.622683,0.090698,0.705412,0.380284,0.766235,0.911661,0.052209,0.398261,0.115420,0.584410,0.250263,0.678473,0.036897,0.320222,0.732811,0.124378,0.487197,0.603410,0.395210,0.118139,0.755617,0.207309,0.839326,0.654648,0.312630,0.780297,0.251412,0.863653,0.938927,0.387793,0.567766,0.622554,0.270126,0.000425,0.512213,0.662080,0.020973,0.708770,0.326954,0.053663,0.367641,0.671891,0.925421,0.067630,0.813049,0.613001,0.137813,0.900515,0.510650,0.648352,0.968607,
    0.332696,0.849883,0.985047,0.178172,0.466262,0.342668,0.697907,0.409344,0.249628,0.153848,0.897480,0.326391,0.688043,0.934347,0.556318,0.136304,0.870843,0.091646,0.534368,0.840545,0.209776,0.827546,0.326102,0.465683,0.761684,0.184477,0.609297,0.186727,0.899271,0.072985,0.803893,0.895379,0.232826,0.530687,0.405849,0.000145,0.815475,0.068777,0.238616,0.692669,0.348406,0.532129,0.185155,0.116119,0.876354,0.145341,0.799438,0.583995,0.339605,0.245657,0.825072,0.929670,0.736981,0.259952,0.837884,0.585586,0.157381,0.991633,0.279340,0.431882,0.728829,0.563618,0.019012,0.235275,
    0.623866,0.579136,0.764360,0.635679,0.008491,0.543168,0.147674,0.936277,0.787725,0.662033,0.452913,0.765456,0.386494,0.008386,0.799225,0.289243,0.602771,0.669516,0.441530,0.349711,0.724022,0.499758,0.158784,0.043992,0.885174,0.446258,0.829841,0.326059,0.517119,0.266376,0.326270,0.562437,0.773697,0.685586,0.748183,0.130415,0.991291,0.577158,0.496555,0.989411,0.096118,0.781843,0.833463,0.690081,0.983449,0.523093,0.887285,0.188435,0.940452,0.421248,0.573523,0.498071,0.105091,0.967978,0.145897,0.458781,0.323515,0.541551,0.749057,0.907308,0.156893,0.356778,0.976238,0.724832,
    0.064496,0.967543,0.360878,0.247452,0.833318,0.903002,0.317271,0.096590,0.538439,0.240938,0.300480,0.922829,0.172180,0.419515,0.321756,0.465349,0.170123,0.256007,0.067862,0.877042,0.009476,0.797544,0.989291,0.385791,0.806907,0.093174,0.985562,0.111614,0.674609,0.448955,0.934346,0.039152,0.122064,0.938762,0.304603,0.487983,0.395216,0.280881,0.140574,0.634616,0.307791,0.615089,0.477444,0.328559,0.412746,0.018009,0.247520,0.497034,0.455685,0.612157,0.735525,0.082124,0.250657,0.418815,0.566975,0.763660,0.045832,0.706503,0.939554,0.054609,0.582650,0.811829,0.135033,0.470728,
    0.813378,0.411689,0.134459,0.690036,0.546548,0.443289,0.082789,0.372029,0.998486,0.571991,0.070289,0.540683,0.732245,0.161035,0.997400,0.888657,0.755950,0.521245,0.975998,0.647457,0.194594,0.581329,0.677660,0.305919,0.578564,0.188047,0.694212,0.850183,0.587091,0.921284,0.703417,0.391267,0.863930,0.238471,0.578770,0.963991,0.646319,0.797428,0.756855,0.874759,0.217444,0.043647,0.936450,0.163793,0.591099,0.810619,0.753552,0.126738,0.721283,0.056710,0.335825,0.862343,0.590835,0.722783,0.901984,0.995479,0.358747,0.074221,0.197321,0.226290,0.443345,0.693105,0.256903,0.533152,
    0.660291,0.867783,0.950962,0.495781,0.296461,0.801776,0.678261,0.833418,0.651319,0.045292,0.706375,0.809717,0.472128,0.097827,0.636589,0.144653,0.785463,0.289433,0.699158,0.316820,0.539184,0.834780,0.160140,0.663833,0.939017,0.473612,0.403507,0.282225,0.053744,0.551346,0.209701,0.468001,0.791380,0.560194,0.032078,0.838381,0.231384,0.355120,0.486742,0.965020,0.011704,0.763065,0.272070,0.801267,0.205367,0.873862,0.423830,0.710888,0.998470,0.662067,0.179771,0.915181,0.150143,0.380200,0.084837,0.207232,0.606457,0.507412,0.882938,0.788611,0.925034,0.024887,0.892394,0.060626,
    0.538742,0.094357,0.639428,0.028814,0.893213,0.212524,0.098964,0.467098,0.177966,0.341314,0.959475,0.220368,0.729375,0.341402,0.565960,0.402917,0.029760,0.919999,0.066210,0.449102,0.721960,0.315117,0.625266,0.049598,0.225692,0.816788,0.911617,0.138074,0.661555,0.373019,0.992746,0.112857,0.350797,0.727586,0.449493,0.103230,0.156696,0.554353,0.047662,0.706788,0.378669,0.582008,0.350141,0.546782,0.096305,0.328749,0.645498,0.144931,0.309822,0.505785,0.789367,0.102438,0.997199,0.286023,0.553062,0.806764,0.140105,0.746121,0.676833,0.165716,0.332957,0.581694,0.221059,0.394474,
    0.294233,0.751737,0.848930,0.340600,0.662059,0.590075,0.418072,0.264213,0.880496,0.625603,0.469097,0.051479,0.974299,0.940790,0.258985,0.839146,0.114656,0.592699,0.471460,0.162069,0.972235,0.524801,0.422549,0.906359,0.784150,0.064757,0.314501,0.516298,0.794378,0.224490,0.744751,0.646617,0.318503,0.810159,0.684730,0.371340,0.865938,0.265419,0.688349,0.108014,0.423719,0.870560,0.279117,0.675697,0.987524,0.165753,0.535717,0.807473,0.617487,0.427659,0.035532,0.621104,0.510451,0.811397,0.968901,0.107382,0.441624,0.264397,0.618658,0.483215,0.880021,0.560240,0.713602,0.130349,
    0.928097,0.526160,0.198088,0.243855,0.865270,0.143916,0.866784,0.564448,0.019156,0.281780,0.420945,0.664808,0.585506,0.498620,0.360507,0.769154,0.527957,0.734161,0.038330,0.642487,0.247257,0.069679,0.721645,0.216817,0.336330,0.633517,0.065528,0.983142,0.171608,0.477362,0.579052,0.011471,0.891890,0.249807,0.182377,0.928249,0.588983,0.439116,0.795104,0.630891,0.975087,0.733867,0.164469,0.891611,0.790193,0.124061,0.414650,0.101501,0.896269,0.890441,0.266998,0.705117,0.303965,0.115723,0.369741,0.659310,0.858161,0.957245,0.342215,0.206083,0.106245,0.999333,0.465643,0.825350,
    0.120012,0.962943,0.560388,0.484681,0.046627,0.773120,0.451855,0.164144,0.517263,0.996877,0.081299,0.190539,0.905568,0.082393,0.164014,0.881194,0.331500,0.154643,0.797958,0.387513,0.552274,0.767170,0.100824,0.953394,0.901216,0.496507,0.095640,0.635923,0.881721,0.305610,0.150232,0.967424,0.485048,0.059609,0.612594,0.827931,0.075335,0.364440,0.998539,0.219005,0.289849,0.515160,0.098050,0.608269,0.374591,0.064740,0.965032,0.351283,0.499465,0.196970,0.944026,0.354317,0.789913,0.932182,0.549722,0.726025,0.193800,0.020896,0.510225,0.682863,0.787043,0.049970,0.291349,0.511649,
    0.060346,0.722895,0.310783,0.669690,0.614221,0.371996,0.892781,0.402817,0.693040,0.849827,0.809801,0.681525,0.372009,0.728523,0.644918,0.417894,0.261207,0.486278,0.046415,0.901999,0.707549,0.471711,0.618922,0.290131,0.711839,0.459272,0.340069,0.810861,0.411874,0.693905,0.817779,0.354822,0.447746,0.673724,0.400401,0.209647,0.528079,0.774799,0.033454,0.468891,0.144670,0.807650,0.292155,0.961628,0.485489,0.727484,0.525640,0.793952,0.819435,0.068243,0.685114,0.517974,0.166335,0.843408,0.040562,0.424979,0.626391,0.939367,0.807905,0.313597,0.455881,0.353106,0.651269,0.799158,
    0.500160,0.270573,0.793579,0.855009,0.951456,0.118547,0.529885,0.600911,0.033125,0.116337,0.450479,0.560921,0.208816,0.794920,0.061281,0.824355,0.946270,0.690331,0.342750,0.112607,0.380187,0.212968,0.029873,0.501637,0.115919,0.982127,0.193952,0.276800,0.944564,0.013908,0.194061,0.729293,0.114980,0.822964,0.578151,0.130862,0.947540,0.290054,0.683114,0.623273,0.853144,0.080611,0.427480,0.861460,0.002924,0.272569,0.151278,0.222292,0.568363,0.435535,0.869292,0.195455,0.457264,0.678308,0.388207,0.126247,0.264171,0.778710,0.113448,0.171802,0.969339,0.523391,0.293690,0.890249,
    0.123534,0.562392,0.025629,0.374978,0.336100,0.239855,0.195103,0.946687,0.795831,0.242968,0.318154,0.929254,0.548529,0.322546,0.262083,0.628622,0.129764,0.242806,0.638302,0.884240,0.734860,0.970135,0.766709,0.194143,0.826698,0.096106,0.587512,0.661054,0.139558,0.751685,0.384301,0.606591,0.946417,0.502551,0.310160,0.487794,0.862599,0.747869,0.442629,0.242580,0.576607,0.927778,0.163800,0.562426,0.760546,0.647848,0.465006,0.008015,0.713741,0.103377,0.638665,0.977068,0.091241,0.289743,0.883866,0.823025,0.550490,0.720012,0.358620,0.054798,0.613217,0.855647,0.223791,0.689988,
    0.846735,0.622380,0.964563,0.466524,0.628313,0.807349,0.717188,0.297712,0.383042,0.657694,0.820996,0.040974,0.290008,0.755194,0.951740,0.436626,0.565561,0.780158,0.294202,0.423034,0.575591,0.098013,0.162294,0.659781,0.876518,0.444641,0.954381,0.732472,0.073585,0.530193,0.856664,0.902908,0.031347,0.231944,0.760410,0.082389,0.672345,0.206750,0.305567,0.969582,0.197509,0.785069,0.469176,0.216217,0.932662,0.127347,0.860816,0.263793,0.960575,0.794486,0.198038,0.343444,0.778548,0.612130,0.141878,0.944298,0.090555,0.341991,0.838391,0.949974,0.320770,0.114980,0.769169,0.995609,
    0.232698,0.347069,0.092945,0.763162,0.182270,0.924508,0.072947,0.493401,0.619133,0.866946,0.082726,0.514610,0.655398,0.852620,0.151391,0.332032,0.024997,0.846749,0.541769,0.160831,0.732998,0.459081,0.336392,0.509798,0.300389,0.260746,0.488957,0.807607,0.305207,0.405504,0.675435,0.253614,0.657399,0.372152,0.788617,0.548279,0.441758,0.936772,0.092716,0.033431,0.537972,0.373911,0.002838,0.276225,0.680371,0.147953,0.566914,0.205195,0.462374,0.074753,0.391531,0.844735,0.233996,0.502845,0.471415,0.286550,0.982946,0.643878,0.455331,0.219840,0.741704,0.547525,0.165557,0.452090,
    0.651799,0.484488,0.836812,0.302018,0.685988,0.364947,0.658947,0.158750,0.401251,0.186620,0.939140,0.415208,0.108912,0.215574,0.558689,0.922072,0.776646,0.653316,0.085655,0.929132,0.273097,0.912886,0.537674,0.944099,0.782915,0.068171,0.586950,0.366085,0.980190,0.103916,0.807808,0.496362,0.325960,0.630437,0.094343,0.893302,0.631660,0.002475,0.825038,0.651604,0.891814,0.399745,0.733081,0.968136,0.600304,0.358830,0.841017,0.936782,0.211507,0.792988,0.551747,0.614105,0.732971,0.026230,0.673821,0.339666,0.502902,0.737743,0.163606,0.584237,0.347830,0.955783,0.148977,0.000004,
    0.353866,0.812696,0.171707,0.577777,0.883014,0.491450,0.041353,0.889775,0.815614,0.566223,0.799267,0.998344,0.522097,0.450168,0.118516,0.574513,0.258738,0.163482,0.864342,0.394178,0.604713,0.682800,0.000576,0.379203,0.130118,0.665321,0.841310,0.594771,0.009027,0.444555,0.200572,0.738626,0.041533,0.962188,0.279606,0.189229,0.370794,0.457711,0.698387,0.163309,0.246578,0.037385,0.816053,0.669260,0.112001,0.413404,0.473476,0.663923,0.511160,0.718387,0.092974,0.158663,0.942379,0.313108,0.910937,0.114657,0.196133,0.874088,0.062424,0.792851,0.411174,0.808907,0.871961,0.736230,
    0.619999,0.930334,0.030334,0.278891,0.790238,0.182435,0.649269,0.274126,0.953614,0.699716,0.246659,0.279477,0.715357,0.876137,0.966845,0.800387,0.439250,0.714040,0.317636,0.774879,0.129706,0.997356,0.770245,0.626372,0.470924,0.894183,0.215971,0.238059,0.743844,0.941998,0.154888,0.990756,0.373993,0.853258,0.522206,0.757186,0.807188,0.057552,0.626495,0.571736,0.855988,0.203734,0.546348,0.501163,0.285266,0.794227,0.076262,0.864475,0.271697,0.990559,0.877370,0.314727,0.712807,0.537446,0.846235,0.607311,0.805130,0.279329,0.435294,0.093856,0.631738,0.257771,0.527475,0.052741,
    0.675363,0.266545,0.972916,0.615394,0.360726,0.445176,0.712625,0.440844,0.211309,0.500888,0.049006,0.428786,0.160662,0.512948,0.350166,0.177616,0.067246,0.942057,0.234623,0.502903,0.043832,0.313638,0.105657,0.236434,0.717435,0.028218,0.422778,0.675758,0.327400,0.607204,0.432997,0.704323,0.532447,0.023798,0.456908,0.110726,0.300172,0.786760,0.453918,0.939326,0.300430,0.705237,0.879986,0.139960,0.899770,0.574750,0.739503,0.408856,0.159581,0.695544,0.106586,0.238238,0.379028,0.070397,0.133858,0.467962,0.991540,0.536867,0.186915,0.931101,0.975801,0.469683,0.386526,0.202691,
    0.949412,0.425013,0.179431,0.740853,0.460453,0.120418,0.003953,0.744419,0.848640,0.326729,0.986960,0.597463,0.088749,0.825919,0.460980,0.052058,0.583076,0.835542,0.144032,0.364644,0.672903,0.532919,0.837250,0.976184,0.494851,0.775563,0.979420,0.505311,0.786254,0.050988,0.900919,0.123143,0.182397,0.630048,0.780113,0.958789,0.164746,0.568983,0.354579,0.113059,0.061817,0.412796,0.538126,0.002091,0.333012,0.956430,0.114265,0.229078,0.636632,0.348494,0.532622,0.778864,0.655555,0.747926,0.030275,0.398143,0.668421,0.746443,0.842514,0.563081,0.296337,0.083375,0.709691,0.316429,
    0.131059,0.726794,0.817266,0.302526,0.237742,0.924370,0.783066,0.350176,0.551115,0.693868,0.396938,0.872384,0.746290,0.302873,0.655500,0.762728,0.405888,0.620453,0.493224,0.808210,0.267429,0.876792,0.394451,0.146042,0.244374,0.617598,0.078260,0.909455,0.325109,0.204964,0.834785,0.569497,0.943428,0.350566,0.476760,0.221272,0.838863,0.687919,0.305757,0.992842,0.923412,0.619193,0.748300,0.274531,0.069110,0.619603,0.463127,0.985651,0.453695,0.864558,0.031080,0.573322,0.494206,0.978479,0.884683,0.255212,0.455148,0.125510,0.733268,0.011170,0.637850,0.800173,0.867850,0.528711,
    0.913002,0.471966,0.021351,0.386046,0.885822,0.643687,0.224583,0.445743,0.135943,0.928313,0.091341,0.657662,0.528007,0.016879,0.866741,0.983244,0.326269,0.115943,0.698924,0.573716,0.055885,0.819249,0.270864,0.734778,0.547297,0.388483,0.307635,0.040617,0.552117,0.284829,0.660943,0.502609,0.334359,0.233683,0.713396,0.150858,0.737030,0.332923,0.083965,0.814526,0.508360,0.072466,0.879713,0.787877,0.937275,0.576172,0.834506,0.585652,0.735841,0.174101,0.924313,0.184670,0.377130,0.827795,0.324821,0.634715,0.490580,0.078251,0.361120,0.166502,0.433538,0.047984,0.211538,0.624993,
    0.564948,0.808222,0.242991,0.677613,0.496680,0.735832,0.840979,0.573506,0.030358,0.457446,0.202910,0.781809,0.174843,0.412122,0.341585,0.158771,0.246270,0.627619,0.061186,0.951208,0.403236,0.173536,0.601370,0.047877,0.129248,0.881212,0.958452,0.685574,0.752241,0.451944,0.124445,0.878975,0.062744,0.807534,0.953472,0.004394,0.500112,0.589556,0.029201,0.211261,0.642113,0.323056,0.464836,0.102105,0.168838,0.340758,0.141514,0.019111,0.330877,0.682740,0.442905,0.060708,0.646755,0.228697,0.173004,0.057961,0.969896,0.844614,0.556239,0.809484,0.977877,0.607426,0.369822,0.095569,
    0.972022,0.584685,0.099446,0.401849,0.015513,0.101154,0.169061,0.366570,0.894159,0.824126,0.607316,0.267376,0.910059,0.603761,0.706728,0.525937,0.803519,0.497693,0.676262,0.872317,0.736373,0.473096,0.373188,0.996446,0.815000,0.558593,0.633667,0.135997,0.882934,0.556455,0.819483,0.472344,0.183215,0.656548,0.577483,0.374841,0.897524,0.290068,0.112590,0.750905,0.477673,0.275798,0.860101,0.555735,0.616751,0.415173,0.578124,0.802371,0.582610,0.073458,0.757890,0.402653,0.991050,0.540482,0.727798,0.901304,0.420986,0.236895,0.721828,0.261108,0.113866,0.904504,0.786798,0.297782,
    0.039189,0.394213,0.640834,0.843331,0.951465,0.312934,0.855175,0.471143,0.649848,0.982188,0.392185,0.552260,0.984979,0.100905,0.856968,0.599418,0.104627,0.969256,0.241302,0.516162,0.017806,0.284409,0.640404,0.096956,0.215874,0.291819,0.430674,0.356708,0.189813,0.976160,0.055953,0.339637,0.549387,0.959513,0.240188,0.125058,0.742525,0.992666,0.879582,0.388102,0.955932,0.667687,0.187488,0.977952,0.255784,0.701281,0.947515,0.852554,0.305328,0.188178,0.016901,0.858810,0.511229,0.295916,0.763449,0.187349,0.618707,0.526581,0.937019,0.766718,0.433100,0.186017,0.726319,0.491626,
    0.846324,0.325657,0.168287,0.682597,0.739191,0.538082,0.048836,0.163657,0.092052,0.002892,0.215287,0.689854,0.417224,0.942879,0.288617,0.402760,0.354371,0.139858,0.762723,0.597206,0.144016,0.778411,0.985118,0.793461,0.530659,0.708091,0.068658,0.788063,0.506380,0.240455,0.675105,0.303444,0.375619,0.480174,0.787217,0.841302,0.206845,0.665837,0.013009,0.524459,0.133018,0.740248,0.095801,0.791379,0.380016,0.196181,0.105669,0.242210,0.453069,0.692740,0.970019,0.139094,0.356939,0.638274,0.076659,0.659132,0.004379,0.083389,0.189605,0.319818,0.573488,0.088061,0.161449,0.693060,
    0.979820,0.525402,0.074225,0.390666,0.462264,0.281077,0.427576,0.704005,0.550335,0.938175,0.496395,0.093493,0.254712,0.569360,0.802344,0.651619,0.015273,0.825952,0.266028,0.442305,0.598726,0.819046,0.659409,0.410357,0.927103,0.896155,0.006197,0.841035,0.145570,0.907464,0.619906,0.800250,0.724490,0.035970,0.406013,0.615803,0.450099,0.283049,0.897545,0.365196,0.180535,0.571468,0.389139,0.861368,0.006539,0.496145,0.795603,0.672239,0.885194,0.518070,0.607008,0.805020,0.231063,0.910189,0.423596,0.812280,0.401095,0.818207,0.675070,0.162017,0.914321,0.382912,0.684774,0.445556,
    0.226559,0.179391,0.935515,0.958794,0.054002,0.808952,0.833299,0.397440,0.762145,0.647691,0.348101,0.828118,0.636514,0.002807,0.179689,0.681687,0.993440,0.533975,0.405821,0.939089,0.164281,0.038532,0.263792,0.587703,0.186348,0.372522,0.280946,0.554067,0.755615,0.414380,0.000619,0.527374,0.106811,0.919251,0.160625,0.038619,0.977898,0.081856,0.506349,0.726097,0.809630,0.017847,0.511919,0.331109,0.717878,0.592662,0.212540,0.555496,0.027634,0.839722,0.298370,0.438411,0.729960,0.276457,0.143141,0.703107,0.293786,0.961622,0.484987,0.640026,0.820193,0.773701,0.263659,0.733865,
    0.007738,0.615543,0.725955,0.204386,0.639484,0.123919,0.942477,0.295081,0.115626,0.981353,0.461191,0.224048,0.947627,0.768941,0.387954,0.279423,0.560351,0.071908,0.632797,0.216173,0.728422,0.331663,0.138837,0.826765,0.151000,0.650198,0.467193,0.696110,0.187107,0.958976,0.153821,0.245203,0.775256,0.320430,0.510379,0.719903,0.587430,0.328253,0.655844,0.266323,0.971345,0.764514,0.908706,0.182787,0.125938,0.989905,0.285869,0.820637,0.393426,0.175138,0.484221,0.098546,0.688122,0.528311,0.835573,0.568532,0.069053,0.768483,0.430273,0.132022,0.317594,0.500676,0.567664,0.876259,
    0.703861,0.568556,0.325360,0.669607,0.402930,0.035916,0.749506,0.497533,0.579054,0.902254,0.290556,0.115347,0.701608,0.481549,0.058537,0.830397,0.756798,0.109376,0.319959,0.878273,0.571355,0.968350,0.478111,0.703161,0.741831,0.048423,0.828833,0.130196,0.291261,0.580427,0.749138,0.451918,0.989828,0.633693,0.261751,0.787432,0.899139,0.803857,0.096073,0.449214,0.238613,0.488051,0.253318,0.773278,0.563871,0.794606,0.050460,0.707036,0.931457,0.077956,0.750605,0.963328,0.200591,0.009560,0.981874,0.372930,0.866196,0.632532,0.231313,0.876902,0.018749,0.942794,0.368920,0.107170,

};


int main(int argc, char** argv)
{
    const int NUM_THREAD = 72;

    const int Width = 1920;
    const int Height = 1080;

    FloatImage image(Width, Height);
    bool end_flag = false;

    // 時間監視スレッドを立てる
    std::thread watcher([&end_flag, &image]() {
        int image_index = 0;
        char buf[256];

        // 開始時間を取得しておく
        auto start = std::chrono::system_clock::now();

        auto tick_start = start;
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 0.1秒眠る

            // 15秒経過を計る
            auto current = std::chrono::system_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(current - tick_start).count() >= 15 * 1000) {
                // 画像出力
                tick_start = current;

                sprintf(buf, "%03d.png", image_index);
                save_image(buf, image);
                std::cout << "Saved: " << buf << std::endl;
                ++image_index;
            }

            // 123秒経過を計る
            if (std::chrono::duration_cast<std::chrono::milliseconds>(current - start).count() >= 123 * 1000) {
                // 画像出力して全部終了じゃい
                end_flag = true;
                save_image("final_image.png", image, true);
                std::cout << "Saved: final_image.png" << std::endl;
                return;
            }
        }
    });

    // workリストを適当にこしらえる
    struct Work
    {
        int begin_x = 0;
        int begin_y = 0;
        int end_x = 0;
        int end_y = 0;

        FloatImage* image = nullptr;
        std::atomic<bool> working = false;

        Work() {}

        Work(const Work& w) noexcept
        {
            begin_x = w.begin_x;
            begin_y = w.begin_y;
            end_x = w.end_x;
            end_y = w.end_y;
            image = w.image;
            working.store(w.working.load());
        }
    };

    // 時間の許す限り処理を続ける
    std::vector<Work> work_list;
    int work_list_index = 0;
    int num_loop = 0;

    static constexpr int BlockWidth = 32;
    static constexpr int BlockHeight = 32;

    for (int begin_y = 0; begin_y < Height; begin_y += BlockHeight) {
        for (int begin_x = 0; begin_x < Width; begin_x += BlockWidth) {
            Work w;

            w.begin_x = begin_x;
            w.begin_y = begin_y;
            w.end_x = hmath::clamp(begin_x + BlockWidth, 0, Width);
            w.end_y = hmath::clamp(begin_y + BlockHeight, 0, Height);
            w.image = &image;

            work_list.push_back(w);
        }
    }

    // スレッドに処理を振り分ける
    std::vector<std::thread> thread_list;
    std::mutex work_list_mutex;

    for (int thread_id = 0; thread_id < NUM_THREAD; ++thread_id) {
        std::thread thread([Width, Height, &end_flag, thread_id, &work_list, &work_list_index, &num_loop, &work_list_mutex]() {
            hetc::set_thread_group(thread_id);

            hmath::Rng rng;
            rng.set_seed(thread_id);

            for (;;) {
                Work* current_work = nullptr;
                {
                    std::lock_guard<std::mutex> lock(work_list_mutex);

                    if (!work_list[work_list_index].working.load()) {
                        current_work = &work_list[work_list_index];
                        work_list[work_list_index].working.store(true);
                    }
                    work_list_index++;
                    if (work_list_index == work_list.size()) {
                        work_list_index = 0;
                        ++num_loop;
                    }
                }

                if (current_work == nullptr) {
                    continue;
                }

                // タスク処理
                for (int iy = current_work->begin_y; iy < current_work->end_y; ++iy) {
                    for (int ix = current_work->begin_x; ix < current_work->end_x; ++ix) {

                        auto& p = (*current_work->image)(ix, iy);
                        auto& vp = (*current_work->image).data2(ix, iy);

                        for (int batch = 0; batch < 1; ++batch) {
                            if (end_flag)
                                return;

                            //float radiance[3];
                            //htest::get_radiance(ix, iy, rng.next(), radiance);

                            //const auto ret = integrator::get_radiance(Width, Height, ix, iy, num_loop % 10 < 10 ? num_loop : rng.next() /* rng.next() */);


                            const int current_seed = (int)(bluenoise[(ix % 64) + (iy % 64) * 64] * 4096 * 64) + num_loop;
                            //const int current_seed = num_loop;// rng.next();

                            /*
                            const auto aspect = (float)Width / Height;
                            const float u = ((float)ix / Width - 0.5f) * aspect;
                            const float v = (float)iy / Height - 0.5f;

                            float dist = sqrt(u * u + v * v) - 0.2f;
                            if (dist < 0)
                                dist = 0;

                            const int current_seed = (int)(bluenoise[(ix % 64) + (iy % 64) * 64] * (2 + 4096 * 16 * dist *  dist)) + num_loop;
                            */
                            const auto ret = integrator::get_radiance(Width, Height, ix, iy, current_seed);

                            if (is_valid(ret)) {
                                const auto dret = hmath::Double3(ret[0], ret[1], ret[2]);
                                p += dret;

                                // 分散計算用
                                const auto lumi = dot(dret, rgb2y);
                                vp[0] += lumi * lumi;

                                current_work->image->samples(ix, iy) += 1;
                            }
                        }
                    }
                }

                // 完了
                current_work->working.store(false);
            }
        });

        thread_list.push_back(std::move(thread));
    }

    for (auto& t : thread_list) {
        t.join();
    }

    watcher.join();

    return 0;
}
#endif