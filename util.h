#ifndef _HMATH_UTIL_
#define _HMATH_UTIL_

#include <cinttypes>
#include <limits>
#include <tuple>

namespace hmath
{
    // [minv, maxv]‚ÉŠÛ‚ß‚é
    template<typename T>
    T clamp(T x, T minv, T maxv)
    {
        if (x <= minv)
            return minv;
        if (maxv <= x)
            return maxv;
        return x;
    }

    // rand
    struct splitmix64
    {
        uint64_t x;

        splitmix64(uint64_t a = 0) : x(a) {}

        uint64_t next() {
            uint64_t z = (x += 0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
            z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
            return z ^ (z >> 31);
        }
    };

    // PCG(64/32)
    // http://www.pcg-random.org/download.html
    // initial_inc from official library
    struct PCG_64_32
    {
        uint64_t state;
        uint64_t inc;

        PCG_64_32(uint64_t initial_state = 0x853c49e6748fea9bULL, uint64_t initial_inc = 0xda3e39cb94b95bdbULL) : state(initial_state), inc(initial_inc) {}

        void set_seed(uint64_t seed)
        {
            splitmix64 s(seed);
            state = s.next();
        }

        uint32_t rotr(uint32_t x, int shift)
        {
            return (x >> shift) | (x << (32 - shift));
        }

        using return_type = uint32_t;
        return_type next()
        {
            auto oldstate = state;
            state = oldstate * 6364136223846793005ULL + (inc | 1);
            uint32_t xorshifted = uint32_t(((oldstate >> 18u) ^ oldstate) >> 27u);
            uint32_t rot = oldstate >> 59u;

            return rotr(xorshifted, rot);
        }

        // [0, 1)
        float next01()
        {
            return (float)(((double)next()) / ((double)std::numeric_limits<uint32_t>::max() + 1));
        }
    };

    using Rng = PCG_64_32;

    template<typename real>
    constexpr real pi()
    {
        return (real)3.14159265358979323846;
    }


    // https://github.com/mitsuba-rei/rt/blob/master/01/rt.cpp#L52
    template<typename V>
    std::tuple<V, V> tangentSpace(const V& n)
    {
        const float s = (float)std::copysign(1, n[2]);
        const float a = -1 / (s + n[2]);
        const float b = n[0]*n[1]*a;
        return {
            V(1 + s * n[0]*n[0]*a,s*b,-s * n[0]),
            V(b,s + n[1]*n[1]*a,-n[1])
        };
    }
}

#endif