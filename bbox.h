#ifndef _BBOX_H_
#define _BBOX_H_

#include "vec3.h"

#include <algorithm>

namespace hrt
{
    using namespace hmath;

    class BBox {
    private:
    public:
        Float3 pmin, pmax;

        BBox() {
            const auto inf = std::numeric_limits<float>::infinity();
            pmin = Float3(inf, inf, inf);
            pmax = -1.0f * pmin;
        }

        BBox(const Float3 &p) : pmin(p), pmax(p) {}

        BBox(const Float3 &p1, const Float3 &p2) {
            pmin = Float3(std::min(p1[0], p2[0]), std::min(p1[1], p2[1]), std::min(p1[2], p2[2]));
            pmax = Float3(std::max(p1[0], p2[0]), std::max(p1[1], p2[1]), std::max(p1[2], p2[2]));
        }

        bool inside(const Float3 &pt) const {
            return (pmin[0] <= pt[0] && pt[0] <= pmax[0] &&
                pmin[1] <= pt[1] && pt[1] <= pmax[1] &&
                pmin[2] <= pt[2] && pt[2] <= pmax[2]);
        }

        Float3 &operator[](int i) {
            if (i == 0)
                return pmin;
            return pmax;
        }
        const Float3 &operator[](int i) const {
            if (i == 0)
                return pmin;
            return pmax;
        }

        void expand(float delta) {
            const Float3 v(delta, delta, delta);
            pmin = pmin - v;
            pmax = pmax + v;
        }

        float surface_area() {
            const Float3 d = pmax - pmin;
            return 2.0f * (d[0] * d[1] + d[0] * d[2] + d[1] * d[2]);
        }
        float volume() {
            const Float3 d = pmax - pmin;
            return d[0] * d[1] * d[2];
        }

        enum LongestAxis {
            AxisX,
            AxisY,
            AxisZ,
        };

        LongestAxis maximum_extent() const {
            const Float3 diag = pmax - pmin;
            if (diag[0] > diag[1] && diag[0] > diag[2])
                return AxisX;
            else if (diag[1] > diag[2])
                return AxisY;
            else
                return AxisZ;
        }

        inline bool check_intersect(const Float3& org, const Float3& dir, float *hitt0, float *hitt1) const {
            float t0 = 0.0, t1 = std::numeric_limits<float>::infinity();
            for (int i = 0; i < 3; ++i) {
                // Update interval for _i_th bounding box slab
                float invRayDir = 1.f / dir[i];
                float tNear = (pmin[i] - org[i]) * invRayDir;
                float tFar = (pmax[i] - org[i]) * invRayDir;

                // Update parametric interval from slab intersection $t$s
                if (tNear > tFar) std::swap(tNear, tFar);
                t0 = tNear > t0 ? tNear : t0;
                t1 = tFar  < t1 ? tFar : t1;
                if (t0 > t1) return false;
            }
            if (hitt0) *hitt0 = t0;
            if (hitt1) *hitt1 = t1;
            return true;
        }
    };
}

#endif
