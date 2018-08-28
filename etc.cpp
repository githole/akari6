#include "etc.h"

// Windows向けのスレッドグループ設定ユーティリティだよ。
// 他OSの人は適当にやってね。

#include <Windows.h>

namespace hetc
{
    void set_thread_group(uint32_t thread_id)
    {
        auto group = thread_id % 2; // 0, 1で、プロセッサグループを指定する。っぽい。環境依存。
        GROUP_AFFINITY mask;
        if (GetNumaNodeProcessorMaskEx(group, &mask))
            SetThreadGroupAffinity(GetCurrentThread(), &mask, nullptr);
    }
}