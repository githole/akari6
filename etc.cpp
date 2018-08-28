#include "etc.h"

// Windows�����̃X���b�h�O���[�v�ݒ胆�[�e�B���e�B����B
// ��OS�̐l�͓K���ɂ���ĂˁB

#include <Windows.h>

namespace hetc
{
    void set_thread_group(uint32_t thread_id)
    {
        auto group = thread_id % 2; // 0, 1�ŁA�v���Z�b�T�O���[�v���w�肷��B���ۂ��B���ˑ��B
        GROUP_AFFINITY mask;
        if (GetNumaNodeProcessorMaskEx(group, &mask))
            SetThreadGroupAffinity(GetCurrentThread(), &mask, nullptr);
    }
}