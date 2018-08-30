#ifndef _HMATH_VEC3_
#define _HMATH_VEC3_

#include <iostream>
#include <type_traits>

namespace hmath
{
    // 超適当ベクトルクラス
    // いちおーテンプレート化しとく
    template<typename Real>
    struct Vec3 final // こういう基本的なクラスは継承したくないよね。
    {
        static_assert(std::is_floating_point<Real>::value, "残念！floatかdoubleだけにしてね。");

        Real v[3] = {}; // 今時の最適化ならいい感じに畳み込まれるっしょー

        Vec3() {}
        explicit Vec3(Real* ptr)
        {
            v[0] = ptr[0];
            v[1] = ptr[1];
            v[2] = ptr[2];
        }
        explicit Vec3(Real x, Real y, Real z)
        {
            v[0] = x;
            v[1] = y;
            v[2] = z;
        }
        ~Vec3() = default;
        Vec3(const Vec3& o) = default;
        Vec3& operator=(const Vec3& o) = default;
        Vec3(Vec3&& o) noexcept = default;
        Vec3& operator=(Vec3&& o) noexcept = default;

        // operator

        // indexのout-of-rangeはめんどいので無視。
        Real operator[](size_t i) const
        {
            return v[i];
        }

        Real& operator[](size_t i)
        {
            return v[i];
        }

        Vec3& operator+=(const Vec3 &o)
        {
            v[0] += o[0]; v[1] += o[1]; v[2] += o[2];
            return *this;
        }

        Vec3& operator-=(const Vec3 &o)
        {
            v[0] -= o[0]; v[1] -= o[1]; v[2] -= o[2];
            return *this;
        }

        Vec3& operator*=(const Real scalar)
        {
            v[0] *= scalar; v[1] *= scalar; v[2] *= scalar;
            return *this;
        }

        Vec3& operator/=(const Real scalar)
        {
            auto inv = 1 / scalar;
            return (*this) *= inv;
        }

        Vec3 operator-() const
        {
            return Vec3(-v[0], -v[1], -v[2]);
        }
    };

    template<typename Real>
    Vec3<Real> operator+(const Vec3<Real>& a, const Vec3<Real>& b)
    {
        auto t{ a };
        t += b;
        return t;
    }

    template<typename Real>
    Vec3<Real> operator-(const Vec3<Real>& a, const Vec3<Real>& b)
    {
        auto t{ a };
        t -= b;
        return t;
    }

    template<typename Real>
    Vec3<Real> operator*(const Vec3<Real>& v, Real scalar)
    {
        auto t{ v };
        t *= scalar;
        return t;
    }

    template<typename Real>
    Vec3<Real> operator*(Real scalar, const Vec3<Real>& v)
    {
        auto t{ v };
        t *= scalar;
        return t;
    }

    template<typename Real>
    Vec3<Real> operator/(const Vec3<Real>& v, Real scalar)
    {
        auto t{ v };
        t /= scalar;
        return t;
    }

    // basic methods

    template<typename Real>
    Real dot(const Vec3<Real>& a, const Vec3<Real>& b)
    {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    template<typename Real>
    Real length2(const Vec3<Real>& v)
    {
        return dot(v, v);
    }

    template<typename Real>
    Real length(const Vec3<Real>& v)
    {
        return sqrt(length2(v));
    }

    template<typename Real>
    Vec3<Real> normalize(const Vec3<Real>& v)
    {
        auto inv_l = 1 / length(v); // zero divはどうすっかねえ？
        return v * inv_l;
    }

    template<typename Real>
    Vec3<Real> cross(const Vec3<Real>& a, const Vec3<Real>& b)
    {
        return Vec3<Real>{
            (a[1] * b[2]) - (a[2] * b[1]),
            (a[2] * b[0]) - (a[0] * b[2]),
            (a[0] * b[1]) - (a[1] * b[0]) };
    }

    // アダマール積
    template<typename Real>
    Vec3<Real> product(const Vec3<Real>& a, const Vec3<Real>& b)
    {
        return Vec3<Real>{ a[0] * b[0], a[1] * b[1], a[2] * b[2] };
    }

    // その他
    template<typename Real>
    std::ostream &operator<<(std::ostream &out, const Vec3<Real> &v)
    {
        out << "<" << v[0] << ", " << v[1] << ", " << v[2] << ">";
        return out;
    }

    template<typename Real>
    bool is_valid(const Vec3<Real>& v)
    {
        for (int i = 0; i < 3; ++i) {
            if (std::isnan(v[i]))
                return false;
        }
        return true;
    }

    template<typename Real>
    Real lerp(Real a, Real b, Real x)
    {
        return (b - a) * x + a;
    }

    template<typename Real>
    Vec3<Real> lerp(Vec3<Real> a, Vec3<Real> b, Real x)
    {
        Vec3<Real> tmp(lerp(a[0], b[0], x), lerp(a[1], b[1], x), lerp(a[2], b[2], x));
        return tmp;
    }




    // Y-up
    template<typename T>
    Vec3<T> polarCoordinateToDirection(T theta, T phi, const Vec3<T>& normal, const Vec3<T>& tangent, const Vec3<T>& binormal)
    {
        return sin(theta) * cos(phi) * tangent + cos(theta) * normal + sin(theta) * sin(phi) * binormal;
    }

    using Float3 = Vec3<float>;
    using Double3 = Vec3<double>;

}



#endif
