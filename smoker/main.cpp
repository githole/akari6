#define NOMINMAX
#include <Windows.h>

#include <vector>
#include <algorithm>
#include <cstdio>

#include <gl/GL.h>
#include <GLFW/glfw3.h>

#include "../util.h"


int main(void)
{
    GLFWwindow* window;

    if (!glfwInit())
        return -1;

    window = glfwCreateWindow(1024, 1024, "Smoker", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    hmath::Rng rng;

    static constexpr int Ny = 128;
    static constexpr int Nx = 128;

    float data[128][128];
    for (int iy = 0; iy < 128; ++iy) {
        for (int ix = 0; ix < 128; ++ix) {
            data[iy][ix] = rng.next01();
        }
    }


    float size = 4.0f;

    uint64_t count = 0;
    while (!glfwWindowShouldClose(window))
    {
        glClear(GL_COLOR_BUFFER_BIT);


        ++count;

        glPointSize(size);
        glBegin(GL_POINTS);
        {
            const float scale = 2.0f;
            for (int iy = 0; iy < Ny; ++iy) {
                for (int ix = 0; ix < Nx; ++ix) {
                    float px = -scale / 2.0 + scale * (float)ix / Nx;
                    float py = -scale / 2.0 + scale * (float)iy / Ny;
                    float pz = 0;
//                    glColor3f(dens[ix][iy], 0, 0);
//                    glVertex2f(ix / (float)N2 - 0.5f, iy / (float)N2 - 0.5f);

                    const int next = (count + 1) % 2;
                    const float value = data[ix][iy];

                    {
                        glColor3f(value, value, value);
                        const float w = 0.005f;
                        const float h = 0.005f;

                        glBegin(GL_TRIANGLE_FAN);

                        glVertex2f(px - w, py - h);
                        glVertex2f(px - w, py + h);
                        glVertex2f(px + w, py + h);
                        glVertex2f(px + w, py - h);

                        glEnd();
                    }
                }
            }
        }
        glEnd();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}