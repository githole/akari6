#ifndef _XYZ_H_
#define _XYZ_H_

namespace hcolor
{
    void get_CIE_XYZ_1931(double wavelength /* [nm] */, double value[3]);

    template<typename real>
    constexpr void CIE_XYZ_to_sRGB(real from[3], real to[3])
    {
        real X = from[0];
        real Y = from[1];
        real Z = from[2];
        to[0] = 3.2410*X - 1.5374*Y - 0.4986*Z;
        to[1] = -0.9692*X + 1.8760*Y + 0.0416*Z;
        to[2] = 0.0556*X - 0.2040*Y + 1.0507*Z;
    }

}


#endif