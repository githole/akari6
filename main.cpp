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

void save_image(const char* filename, FloatImage& image)
{
    const auto Width = image.width();
    const auto Height = image.height();
    uint64_t total = 0;
    std::vector<uint8_t> tonemapped_image(Width * Height * 3);

    std::vector<float> fdata(Width * Height * 3);

    fmath::PowGenerator degamma(1.0f / 2.2f);

    for (int iy = 0; iy < Height; ++iy) {
        for (int ix = 0; ix < Width; ++ix) {
            auto index = ix + iy * Width;
            const double N = image.samples(ix, iy);
            auto p = image(ix, iy) / N;
            total += image.samples(ix, iy);

            tonemapped_image[index * 3 + 0] = (uint8_t)(degamma.get(hmath::clamp((float)p[0], 0.0f, 1.0f)) * 255);
            tonemapped_image[index * 3 + 1] = (uint8_t)(degamma.get(hmath::clamp((float)p[1], 0.0f, 1.0f)) * 255);
            tonemapped_image[index * 3 + 2] = (uint8_t)(degamma.get(hmath::clamp((float)p[2], 0.0f, 1.0f)) * 255);

            //const double lumi = dot(p, rgb2y);
            //p[0] = p[1] = p[2] = image.data2(ix, iy)[0] / (N - 1) - (N / (N - 1)) * lumi * lumi;

            fdata[index * 3 + 0] = (float)p[0];
            fdata[index * 3 + 1] = (float)p[1];
            fdata[index * 3 + 2] = (float)p[2];
        }
    }
    double average = (double)total / (Width * Height);
    std::cout << "Average: " << average << " samples/pixel" << std::endl;
    stbi_write_png(filename, (int)Width, (int)Height, 3, tonemapped_image.data(), (int)(Width * sizeof(uint8_t) * 3));

    hhdr::save("hoge.hdr", fdata.data(), (int)Width, (int)Height, false);
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

    constexpr float MieScale = 1;
    constexpr float scattering_Mie_base = 2.5f; /* [1/m] */
    constexpr float H_Mie = 10; /* [m] */

    template<typename Vec3>
    struct TransmittanceCoeffEvaluator
    {
    private:
        float majorant_;

        static constexpr float a = 0.3f;
        const float cosa = cos(a);
        const float sina = sin(a);

        template<bool majorant = false>
        float scattering_Mie(Vec3 p) const
        {
            float scale = 1;

            /*
            if (dot(normalize(hmath::Float3(0.3, -0.7, 0.2)), p) < 0)
                scale = 0.01f;
            */

            /*
            int iu = p[0] / 5.0f;
            int iv = p[2] / 5.0f;

            if (((iu % 2) + (iv % 2)) % 2) {
                scale = 0.01f;
            }
            */

            scale = 0.01f;
            const float h = p[1];


            if (abs(p[0]) < 3.0f && abs(p[1]) < 2.0f && p[1] > 1.0f)
                scale = 1.0f;

            auto k = p - hmath::Float3(-8, 8, -8);
#if 0
            auto tx = k[0];
            auto ty = k[1];

            k[0] = cosa * tx - sina * ty;
            k[1] = sina * tx + cosa * ty;
#endif
            if (abs(k[2]) < 3.0f && abs(k[1]) < 5.0f)
                scale = 1.0f;

            k = p - hmath::Float3(0, 9, 0);
            if (abs(k[0]) < 4.0f && abs(k[1]) < 3.0f && k[1] > 1.0f)
                scale = 1.0f;

            k = p - hmath::Float3(-1, 0, 8);
            if (abs(k[0]) < 2.0f && abs(k[2]) < 2.0f)
                scale = 1.0f;

            if (majorant)
                scale = 1;

            return scale * MieScale * scattering_Mie_base * /*exp(-h / H_Mie)*/ fmath::exp(-h / H_Mie);
        }

        template<bool majorant = false>
        float absorption_Mie(Vec3 p) const
        {
            return scattering_Mie<majorant>(p) * 0.0001f;
        }

    public:
        Vec3 org;
        Vec3 dir;
        float bnd;

        TransmittanceCoeffEvaluator(const Vec3& o, const Vec3& d, float b) :
            org(o), dir(d), bnd(b)
        {
            majorant_ = (scattering_Mie<true>({}) + absorption_Mie<true>({}));
        }

        float majorant() const
        {
            return majorant_;
        }

        float bound() const
        {
            return bnd;
        }

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

        float operator()(float t) const
        {
            const auto pos = org + t * dir;
            return scattering_Mie(pos) + absorption_Mie(pos);
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

        if (length < 5)
            return {};

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

        Float3 camera_up(0, 1, 0);

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
        const float k1 = -0.2f;
        const float new_U = U * (1.0f + k1 * r);
        const float new_V = V * (1.0f + k1 * r);
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
            auto free_path = volume::delta_tracking<float>(coeff, rng);
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
        }

#if 0
        Float3 coeff(0, 0, 0);
        coeff[ch] = 1;
        contribution = product(contribution, coeff);
        return contribution;
#endif
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
int main(int argc, char** argv)
{
    const int NUM_THREAD = 72;

    const int Width = 1280;
    const int Height = 768;

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
            if (std::chrono::duration_cast<std::chrono::milliseconds>(current - start).count() >= 10 * 1000) {
                // 画像出力して全部終了じゃい
                end_flag = true;
                save_image("final_image.png", image);
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

                            const auto ret = integrator::get_radiance(Width, Height, ix, iy, rng.next());

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