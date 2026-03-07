#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <cmath>
#include <cstring>

static void lstm_gemv_f16(
    const __fp16* x,
    const __fp16* W,
    __fp16* out,
    size_t K,
    size_t N
) {
    const size_t k8 = (K / 8) * 8;
    const size_t N8 = (N / 8) * 8;

    for (size_t n = 0; n < N8; n += 8) {
        float32x4_t acc_lo = vdupq_n_f32(0.0f);
        float32x4_t acc_hi = vdupq_n_f32(0.0f);

        for (size_t k = 0; k < k8; k += 8) {
            float16x8_t xv = vld1q_f16(x + k);

            float16x8_t w0 = vld1q_f16(W + (n + 0) * K + k);
            float16x8_t w1 = vld1q_f16(W + (n + 1) * K + k);
            float16x8_t w2 = vld1q_f16(W + (n + 2) * K + k);
            float16x8_t w3 = vld1q_f16(W + (n + 3) * K + k);
            float16x8_t w4 = vld1q_f16(W + (n + 4) * K + k);
            float16x8_t w5 = vld1q_f16(W + (n + 5) * K + k);
            float16x8_t w6 = vld1q_f16(W + (n + 6) * K + k);
            float16x8_t w7 = vld1q_f16(W + (n + 7) * K + k);

            float16x8_t p0 = vmulq_f16(xv, w0);
            float16x8_t p1 = vmulq_f16(xv, w1);
            float16x8_t p2 = vmulq_f16(xv, w2);
            float16x8_t p3 = vmulq_f16(xv, w3);
            float16x8_t p4 = vmulq_f16(xv, w4);
            float16x8_t p5 = vmulq_f16(xv, w5);
            float16x8_t p6 = vmulq_f16(xv, w6);
            float16x8_t p7 = vmulq_f16(xv, w7);

            float16x4_t r0 = vadd_f16(vget_low_f16(p0), vget_high_f16(p0));
            float16x4_t r1 = vadd_f16(vget_low_f16(p1), vget_high_f16(p1));
            float16x4_t r2 = vadd_f16(vget_low_f16(p2), vget_high_f16(p2));
            float16x4_t r3 = vadd_f16(vget_low_f16(p3), vget_high_f16(p3));
            float16x4_t r4 = vadd_f16(vget_low_f16(p4), vget_high_f16(p4));
            float16x4_t r5 = vadd_f16(vget_low_f16(p5), vget_high_f16(p5));
            float16x4_t r6 = vadd_f16(vget_low_f16(p6), vget_high_f16(p6));
            float16x4_t r7 = vadd_f16(vget_low_f16(p7), vget_high_f16(p7));

            float16x4_t s01 = vpadd_f16(r0, r1);
            float16x4_t s23 = vpadd_f16(r2, r3);
            float16x4_t s45 = vpadd_f16(r4, r5);
            float16x4_t s67 = vpadd_f16(r6, r7);

            float16x4_t sum03 = vpadd_f16(s01, s23);
            float16x4_t sum47 = vpadd_f16(s45, s67);

            acc_lo = vaddq_f32(acc_lo, vcvt_f32_f16(sum03));
            acc_hi = vaddq_f32(acc_hi, vcvt_f32_f16(sum47));
        }

        for (size_t k = k8; k < K; ++k) {
            float xv = (float)x[k];
            for (size_t di = 0; di < 8; ++di) {
                float wv = (float)W[(n + di) * K + k];
                float* acc_ptr = (di < 4) ? (float*)&acc_lo + di : (float*)&acc_hi + (di - 4);
                *acc_ptr += xv * wv;
            }
        }

        vst1_f16(out + n, vcvt_f16_f32(acc_lo));
        vst1_f16(out + n + 4, vcvt_f16_f32(acc_hi));
    }

    for (size_t n = N8; n < N; ++n) {
        float acc = 0.0f;
        for (size_t k = 0; k < K; ++k) {
            acc += (float)x[k] * (float)W[n * K + k];
        }
        out[n] = (__fp16)acc;
    }
}

void cactus_lstm_cell_f16(
    const __fp16* x_input,
    const __fp16* h_prev,
    const __fp16* c_prev,
    const __fp16* weight_ih,
    const __fp16* weight_hh,
    const __fp16* bias_ih,
    const __fp16* bias_hh,
    __fp16* h_new,
    __fp16* c_new,
    size_t batch_size,
    size_t input_size,
    size_t hidden_size
) {
    constexpr size_t SIMD_WIDTH = 8;
    const size_t gate_size = 4 * hidden_size;

    alignas(16) __fp16 gates_ih[2048];
    alignas(16) __fp16 gates_hh[2048];

    const size_t simd_end = (hidden_size / SIMD_WIDTH) * SIMD_WIDTH;

    for (size_t b = 0; b < batch_size; ++b) {
        const size_t hidden_offset = b * hidden_size;

        lstm_gemv_f16(x_input + b * input_size, weight_ih, gates_ih, input_size, gate_size);
        lstm_gemv_f16(h_prev + hidden_offset, weight_hh, gates_hh, hidden_size, gate_size);

        const float32x4_t one = vdupq_n_f32(1.0f);

        for (size_t h = 0; h < simd_end; h += SIMD_WIDTH) {
            float16x8_t i_gate = vaddq_f16(vaddq_f16(vld1q_f16(&gates_ih[h]),
                                                       vld1q_f16(&gates_hh[h])),
                                             vaddq_f16(vld1q_f16(&bias_ih[h]),
                                                       vld1q_f16(&bias_hh[h])));

            float16x8_t f_gate = vaddq_f16(vaddq_f16(vld1q_f16(&gates_ih[hidden_size + h]),
                                                       vld1q_f16(&gates_hh[hidden_size + h])),
                                             vaddq_f16(vld1q_f16(&bias_ih[hidden_size + h]),
                                                       vld1q_f16(&bias_hh[hidden_size + h])));

            float16x8_t g_gate = vaddq_f16(vaddq_f16(vld1q_f16(&gates_ih[2 * hidden_size + h]),
                                                       vld1q_f16(&gates_hh[2 * hidden_size + h])),
                                             vaddq_f16(vld1q_f16(&bias_ih[2 * hidden_size + h]),
                                                       vld1q_f16(&bias_hh[2 * hidden_size + h])));

            float16x8_t o_gate = vaddq_f16(vaddq_f16(vld1q_f16(&gates_ih[3 * hidden_size + h]),
                                                       vld1q_f16(&gates_hh[3 * hidden_size + h])),
                                             vaddq_f16(vld1q_f16(&bias_ih[3 * hidden_size + h]),
                                                       vld1q_f16(&bias_hh[3 * hidden_size + h])));

            float32x4_t i_low = vcvt_f32_f16(vget_low_f16(i_gate));
            float32x4_t i_high = vcvt_f32_f16(vget_high_f16(i_gate));
            float16x8_t i_act = vcombine_f16(vcvt_f16_f32(vdivq_f32(one, vaddq_f32(one, fast_exp_f32x4(vnegq_f32(i_low))))),
                                               vcvt_f16_f32(vdivq_f32(one, vaddq_f32(one, fast_exp_f32x4(vnegq_f32(i_high))))));

            float32x4_t f_low = vcvt_f32_f16(vget_low_f16(f_gate));
            float32x4_t f_high = vcvt_f32_f16(vget_high_f16(f_gate));
            float16x8_t f_act = vcombine_f16(vcvt_f16_f32(vdivq_f32(one, vaddq_f32(one, fast_exp_f32x4(vnegq_f32(f_low))))),
                                               vcvt_f16_f32(vdivq_f32(one, vaddq_f32(one, fast_exp_f32x4(vnegq_f32(f_high))))));

            float32x4_t g_low = vcvt_f32_f16(vget_low_f16(g_gate));
            float32x4_t g_high = vcvt_f32_f16(vget_high_f16(g_gate));
            float16x8_t g_act = vcombine_f16(vcvt_f16_f32(fast_tanh_f32x4(g_low)),
                                               vcvt_f16_f32(fast_tanh_f32x4(g_high)));

            float32x4_t o_low = vcvt_f32_f16(vget_low_f16(o_gate));
            float32x4_t o_high = vcvt_f32_f16(vget_high_f16(o_gate));
            float16x8_t o_act = vcombine_f16(vcvt_f16_f32(vdivq_f32(one, vaddq_f32(one, fast_exp_f32x4(vnegq_f32(o_low))))),
                                               vcvt_f16_f32(vdivq_f32(one, vaddq_f32(one, fast_exp_f32x4(vnegq_f32(o_high))))));

            float16x8_t c_prev_vec = vld1q_f16(&c_prev[hidden_offset + h]);
            float16x8_t c_update = vfmaq_f16(vmulq_f16(f_act, c_prev_vec), i_act, g_act);
            vst1q_f16(&c_new[hidden_offset + h], c_update);

            float32x4_t c_low = vcvt_f32_f16(vget_low_f16(c_update));
            float32x4_t c_high = vcvt_f32_f16(vget_high_f16(c_update));
            float16x8_t c_tanh = vcombine_f16(vcvt_f16_f32(fast_tanh_f32x4(c_low)),
                                                vcvt_f16_f32(fast_tanh_f32x4(c_high)));
            vst1q_f16(&h_new[hidden_offset + h], vmulq_f16(o_act, c_tanh));
        }

        for (size_t h = simd_end; h < hidden_size; ++h) {
            float i_gate_val = static_cast<float>(gates_ih[h] + gates_hh[h] + bias_ih[h] + bias_hh[h]);
            float f_gate_val = static_cast<float>(gates_ih[hidden_size + h] + gates_hh[hidden_size + h] +
                                                   bias_ih[hidden_size + h] + bias_hh[hidden_size + h]);
            float g_gate_val = static_cast<float>(gates_ih[2 * hidden_size + h] + gates_hh[2 * hidden_size + h] +
                                                   bias_ih[2 * hidden_size + h] + bias_hh[2 * hidden_size + h]);
            float o_gate_val = static_cast<float>(gates_ih[3 * hidden_size + h] + gates_hh[3 * hidden_size + h] +
                                                   bias_ih[3 * hidden_size + h] + bias_hh[3 * hidden_size + h]);

            float i_act = 1.0f / (1.0f + expf(-i_gate_val));
            float f_act = 1.0f / (1.0f + expf(-f_gate_val));
            float g_act = tanhf(g_gate_val);
            float o_act = 1.0f / (1.0f + expf(-o_gate_val));

            float c_val = f_act * static_cast<float>(c_prev[hidden_offset + h]) + i_act * g_act;
            c_new[hidden_offset + h] = static_cast<__fp16>(c_val);
            h_new[hidden_offset + h] = static_cast<__fp16>(o_act * tanhf(c_val));
        }
    }
}
