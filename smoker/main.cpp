#define NOMINMAX
#include <Windows.h>

#include <vector>
#include <algorithm>
#include <cstdio>

#include <gl/GL.h>
#include <GLFW/glfw3.h>

#include "../util.h"
#include "../vec3.h"


static constexpr int H = 64;
static constexpr int W = 64;

static constexpr float sigma_i = 2.1f;
static constexpr float sigma_s = 1.0f;

float Comp(hmath::Float3* image, int ix, int iy)
{
    float sum[H] = {};


    // small kernel approx.
//#pragma omp parallel for schedule(dynamic, 1) num_threads(7)
    for (int oy = -9; oy <= 9; ++oy) {
        for (int ox = -9; ox <= 9; ++ox) {
            int sx = ix + ox;
            if (sx < 0)
                sx += W;
            if (sx >= W)
                sx -= W;

            int sy = iy + oy;
            if (sy < 0)
                sy += H;
            if (sy >= H)
                sy -= H;

            float dx = abs(ix - sx);
            if (dx > W / 2)
                dx = W - dx;

            float dy = abs(iy - sy);
            if (dy > H / 2)
                dy = H - dy;
            const float a =
                (dx * dx + dy * dy) / (sigma_i * sigma_i);

            const float b =
                sqrt(abs(image[ix + iy * W][0] - image[sx + sy * W][0])) / (sigma_s * sigma_s);

            sum[sy] += exp(-a - b);
        }
    }

    /*
    // full size kernel
    #pragma omp parallel for schedule(dynamic, 1) num_threads(7)
    for (int sy = 0; sy < H; ++sy) {
    for (int sx = 0; sx < W; ++sx) {
    float dx = abs(ix - sx);
    if (dx > W / 2)
    dx = W - dx;
    float dy = abs(iy - sy);
    if (dy > H / 2)
    dy = H - dy;
    const float a =
    (dx * dx + dy * dy) / (sigma_i * sigma_i);

    const float b =
    sqrt(abs(image.at(ix, iy).x - image.at(sx, sy).x)) / (sigma_s * sigma_s);
    sum[sy] += exp(-a - b);
    }
    }
    */

    float total = 0;
    for (int sy = 0; sy < H; ++sy)
        total += sum[sy];
    return total;
}

void Swap(hmath::Float3* image, int px, int py, int qx, int qy)
{
    auto tmp = image[px + py * W];
    image[px + py * W] = image[qx + qy * W];
    image[qx + qy * W] = tmp;
}


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
    rng.set_seed(10);


    // init
    hmath::Float3 data[H * W];
    for (int iy = 0; iy < H; ++iy) {
        for (int ix = 0; ix < W; ++ix) {
            const float r = rng.next01();
            data[ix + iy * W] = hmath::Float3(r, 0, 0);
        }
    }

    float energy[H * W] = {};
    for (int iy = 0; iy < H; ++iy) {
        for (int ix = 0; ix < W; ++ix) {
            energy[ix + iy * W] = Comp(data, ix, iy);
        }
    }

    float size = 16.0f;

    uint64_t count = 0;
    while (!glfwWindowShouldClose(window))
    {
        glClear(GL_COLOR_BUFFER_BIT);

        // ˆ—

        for (int i = 0; i < 512; ++i)
        {
            float current_energy = 0;

            for (int iy = 0; iy < H; ++iy) {
                for (int ix = 0; ix < W; ++ix) {
                    current_energy += energy[ix + iy * W];
                }
            }
            // printf("[%f]", current_energy);


            const int px = rng.next() % W;
            const int py = rng.next() % H;
            const int qx = rng.next() % W;
            const int qy = rng.next() % H;

            float next_energy = current_energy;
            next_energy -= energy[py * W + px];
            next_energy -= energy[qy * W + qx];

            // test swap
            Swap(data, px, py, qx, qy);

            const float e0 = Comp(data, px, py);
            const float e1 = Comp(data, qx, qy);

            next_energy += (e0 + e1);

            if (next_energy < current_energy) {
                energy[py * W + px] = e0;
                energy[qy * W + qx] = e1;
                continue;
            }

            // recover
            Swap(data, px, py, qx, qy);
        }

        ++count;

        if (glfwGetKey(window, GLFW_KEY_A)) {
            for (int i = 0; i < H * W; ++i) {
                printf("%f,", data[i][0]);
                if ((i + 1) % W == 0)
                    printf("\n");
            }
        }

        glPointSize(size);
        glBegin(GL_POINTS);
        {
            const float scale = 2.0f;
            for (int iy = 0; iy < H; ++iy) {
                for (int ix = 0; ix < W; ++ix) {
                    float px = -scale / 2.0 + scale * (float)ix / W;
                    float py = -scale / 2.0 + scale * (float)iy / H;
                    float pz = 0;
//                    glColor3f(dens[ix][iy], 0, 0);
//                    glVertex2f(ix / (float)N2 - 0.5f, iy / (float)N2 - 0.5f);

                    const int next = (count + 1) % 2;
                    const float value = data[ix + iy * W][0];

                    {
                        glColor3f(value, value, value);
                        const float w = 0.005f * 1.0f;
                        const float h = 0.005f * 1.0f;

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