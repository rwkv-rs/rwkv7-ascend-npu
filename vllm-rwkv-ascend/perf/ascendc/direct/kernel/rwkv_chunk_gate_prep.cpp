#include "kernel_operator.h"

using namespace AscendC;

// Build centered DPLR chunk factors and state-passing scalars directly from
// token-major fp16 vectors.  One block owns one (batch, chunk, head).
class RwkvChunkGatePrepDirectKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR log_decay, GM_ADDR k, GM_ADDR v, GM_ADDR kk, GM_ADDR a,
        GM_ADDR r, GM_ADDR qg, GM_ADDR kg, GM_ADDR ag, GM_ADDR bg,
        GM_ADDR v_chunk, GM_ADDR q_state, GM_ADDR state_keys,
        GM_ADDR end_decay, GM_ADDR offset_exp, uint32_t tokens,
        uint32_t heads, uint32_t width, uint32_t chunk_size,
        uint32_t chunks, uint32_t output_bf16) {
        tokens_ = tokens;
        heads_ = heads;
        width_ = width;
        chunk_size_ = chunk_size;
        chunks_ = chunks;
        output_bf16_ = output_bf16 != 0;
        const uint32_t group = GetBlockIdx();
        head_ = group % heads_;
        chunk_ = (group / heads_) % chunks_;
        batch_ = group / (heads_ * chunks_);
        group_ = group;
        log_decay_gm_.SetGlobalBuffer((__gm__ half*)log_decay);
        k_gm_.SetGlobalBuffer((__gm__ half*)k);
        v_gm_.SetGlobalBuffer((__gm__ half*)v);
        kk_gm_.SetGlobalBuffer((__gm__ half*)kk);
        a_gm_.SetGlobalBuffer((__gm__ half*)a);
        r_gm_.SetGlobalBuffer((__gm__ half*)r);
        const uint32_t group_elements = chunk_size_ * width_;
        qg_gm_.SetGlobalBuffer((__gm__ float*)qg + group_ * group_elements);
        kg_gm_.SetGlobalBuffer((__gm__ float*)kg + group_ * group_elements);
        ag_gm_.SetGlobalBuffer((__gm__ float*)ag + group_ * group_elements);
        bg_gm_.SetGlobalBuffer((__gm__ float*)bg + group_ * group_elements);
        v_chunk_gm_.SetGlobalBuffer(
            (__gm__ float*)v_chunk + group_ * group_elements);
        q_state_gm_.SetGlobalBuffer(
            (__gm__ float*)q_state + group_ * group_elements);
        state_keys_gm_.SetGlobalBuffer(
            (__gm__ float*)state_keys + group_ * 2 * group_elements);
        qg_bf16_gm_.SetGlobalBuffer(
            (__gm__ bfloat16_t*)qg + group_ * group_elements);
        kg_bf16_gm_.SetGlobalBuffer(
            (__gm__ bfloat16_t*)kg + group_ * group_elements);
        ag_bf16_gm_.SetGlobalBuffer(
            (__gm__ bfloat16_t*)ag + group_ * group_elements);
        bg_bf16_gm_.SetGlobalBuffer(
            (__gm__ bfloat16_t*)bg + group_ * group_elements);
        v_chunk_bf16_gm_.SetGlobalBuffer(
            (__gm__ bfloat16_t*)v_chunk + group_ * group_elements);
        q_state_bf16_gm_.SetGlobalBuffer(
            (__gm__ bfloat16_t*)q_state + group_ * group_elements);
        state_keys_bf16_gm_.SetGlobalBuffer(
            (__gm__ bfloat16_t*)state_keys + group_ * 2 * group_elements);
        end_decay_gm_.SetGlobalBuffer(
            (__gm__ float*)end_decay + group_ * width_);
        offset_exp_gm_.SetGlobalBuffer(
            (__gm__ float*)offset_exp + group_ * width_);

        const uint32_t half_bytes = width_ * sizeof(half);
        const uint32_t float_bytes = width_ * sizeof(float);
        pipe_.InitBuffer(input_buffer_, 12 * half_bytes);
        pipe_.InitBuffer(float_buffer_, 15 * float_bytes);
        pipe_.InitBuffer(output_buffer_, 16 * float_bytes);
        pipe_.InitBuffer(output_bf16_buffer_, 16 * width_ * sizeof(bfloat16_t));
        pipe_.InitBuffer(
            inclusive_buffer_, chunk_size_ * width_ * sizeof(float));
        for (uint32_t bank = 0; bank < 2; ++bank) {
            input_ready_[bank] = static_cast<event_t>(
                pipe_.AllocEventID<HardEvent::MTE2_V>());
            input_done_[bank] = static_cast<event_t>(
                pipe_.AllocEventID<HardEvent::V_MTE2>());
        }
        for (uint32_t bank = 0; bank < 2; ++bank) {
            output_ready_[bank] = static_cast<event_t>(
                pipe_.AllocEventID<HardEvent::V_MTE3>());
            output_done_[bank] = static_cast<event_t>(
                pipe_.AllocEventID<HardEvent::MTE3_V>());
        }
    }

    __aicore__ inline uint32_t InputOffset(uint32_t local_token) const {
        const uint32_t token = chunk_ * chunk_size_ + local_token;
        return ((batch_ * tokens_ + token) * heads_ + head_) * width_;
    }

    __aicore__ inline void BeginInput(uint32_t bank, bool reused) {
        if (reused) {
            WaitFlag<HardEvent::V_MTE2>(input_done_[bank]);
        }
    }

    __aicore__ inline void WaitInput(uint32_t bank) {
        SetFlag<HardEvent::MTE2_V>(input_ready_[bank]);
        WaitFlag<HardEvent::MTE2_V>(input_ready_[bank]);
    }

    __aicore__ inline void ReleaseInput(uint32_t bank) {
        SetFlag<HardEvent::V_MTE2>(input_done_[bank]);
    }

    __aicore__ inline void FlushInputs() {
        for (uint32_t bank = 0; bank < 2 && bank < chunk_size_; ++bank) {
            WaitFlag<HardEvent::V_MTE2>(input_done_[bank]);
        }
    }

    __aicore__ inline void WaitOutput(uint32_t bank) {
        WaitFlag<HardEvent::MTE3_V>(output_done_[bank]);
    }

    __aicore__ inline void BeginOutput(uint32_t bank) {
        SetFlag<HardEvent::V_MTE3>(output_ready_[bank]);
        WaitFlag<HardEvent::V_MTE3>(output_ready_[bank]);
    }

    __aicore__ inline void EndOutput(uint32_t bank) {
        SetFlag<HardEvent::MTE3_V>(output_done_[bank]);
    }

    __aicore__ inline void Process() {
        auto input_base = input_buffer_.Get<half>();
        auto inclusive_history = inclusive_buffer_.Get<float>();
        auto work = float_buffer_.Get<float>();
        auto cumulative = work;
        auto offset = work[width_];
        auto last = work[2 * width_];
        auto log_float = work[3 * width_];
        auto key_float = work[4 * width_];
        auto value_float = work[5 * width_];
        auto kk_float = work[6 * width_];
        auto r_float = work[8 * width_];
        auto positive = work[9 * width_];
        auto negative = work[10 * width_];
        auto exclusive_scale = work[11 * width_];
        auto end_relative = work[12 * width_];
        auto temp = work[13 * width_];
        auto beta = work[14 * width_];
        auto output_base = output_buffer_.Get<float>();
        auto output_bf16_base = output_bf16_buffer_.Get<bfloat16_t>();
        auto out = output_base;

        Duplicate(cumulative, 0.0f, width_);
        for (uint32_t token = 0; token < chunk_size_; ++token) {
            const uint32_t bank = token & 1;
            auto log_half = input_base[bank * 6 * width_];
            BeginInput(bank, token >= 2);
            DataCopy(log_half, log_decay_gm_[InputOffset(token)], width_);
            WaitInput(bank);
            Cast(log_float, log_half, RoundMode::CAST_NONE, width_);
            PipeBarrier<PIPE_V>();
            ReleaseInput(bank);
            Add(cumulative, cumulative, log_float, width_);
            PipeBarrier<PIPE_V>();
            Adds(
                inclusive_history[token * width_], cumulative, 0.0f,
                width_);
            if (token == chunk_size_ / 2) {
                Adds(offset, cumulative, 0.0f, width_);
            }
            PipeBarrier<PIPE_V>();
        }
        FlushInputs();
        Adds(last, cumulative, 0.0f, width_);
        PipeBarrier<PIPE_V>();
        Exp(out, offset, 2 * width_);
        PipeBarrier<PIPE_V>();
        Adds(temp, out, 0.0f, width_);
        PipeBarrier<PIPE_V>();
        BeginOutput(0);
        DataCopy(offset_exp_gm_, out, width_);
        DataCopy(end_decay_gm_, out[width_], width_);
        EndOutput(0);
        WaitOutput(0);

        for (uint32_t token = 0; token < chunk_size_; ++token) {
            const uint32_t bank = token & 1;
            auto inputs = input_base[bank * 6 * width_];
            auto key_half = inputs;
            auto value_half = inputs[width_];
            auto kk_half = inputs[2 * width_];
            auto a_half = inputs[3 * width_];
            auto r_half = inputs[4 * width_];
            auto beta_half = inputs[5 * width_];
            auto token_out = output_base[bank * 8 * width_];
            if (token >= 2) {
                WaitOutput(bank);
            }
            const uint32_t input_offset = InputOffset(token);
            BeginInput(bank, token >= 2);
            DataCopy(key_half, k_gm_[input_offset], width_);
            DataCopy(value_half, v_gm_[input_offset], width_);
            DataCopy(kk_half, kk_gm_[input_offset], width_);
            DataCopy(a_half, a_gm_[input_offset], width_);
            DataCopy(r_half, r_gm_[input_offset], width_);
            WaitInput(bank);
            Cast(key_float, key_half, RoundMode::CAST_NONE, width_);
            Cast(value_float, value_half, RoundMode::CAST_NONE, width_);
            Cast(kk_float, kk_half, RoundMode::CAST_NONE, width_);
            Cast(r_float, r_half, RoundMode::CAST_NONE, width_);
            Mul(beta_half, kk_half, a_half, width_);
            PipeBarrier<PIPE_V>();
            Cast(beta, beta_half, RoundMode::CAST_NONE, width_);
            PipeBarrier<PIPE_V>();
            ReleaseInput(bank);

            auto inclusive = inclusive_history[token * width_];
            Sub(positive, inclusive, offset, width_);
            Sub(negative, offset, inclusive, width_);
            if (token == 0) {
                Muls(exclusive_scale, offset, -1.0f, width_);
            } else {
                Sub(
                    exclusive_scale,
                    inclusive_history[(token - 1) * width_], offset,
                    width_);
            }
            Sub(end_relative, last, inclusive, width_);
            PipeBarrier<PIPE_V>();
            Exp(positive, positive, 4 * width_);
            PipeBarrier<PIPE_V>();

            Mul(token_out, r_float, positive, width_);
            Mul(token_out[width_], key_float, negative, width_);
            Mul(token_out[2 * width_], kk_float, exclusive_scale, width_);
            Adds(token_out[4 * width_], value_float, 0.0f, width_);
            PipeBarrier<PIPE_V>();
            Muls(
                token_out[2 * width_], token_out[2 * width_], -1.0f,
                width_);
            Mul(token_out[3 * width_], beta, negative, width_);
            Mul(token_out[5 * width_], token_out, temp, width_);
            Mul(token_out[6 * width_], key_float, end_relative, width_);
            Mul(token_out[7 * width_], beta, end_relative, width_);
            PipeBarrier<PIPE_V>();

            const uint32_t output_offset = token * width_;
            if (output_bf16_) {
                auto token_out_bf16 = output_bf16_base[bank * 8 * width_];
                Cast(
                    token_out_bf16, token_out, RoundMode::CAST_RINT,
                    8 * width_);
                PipeBarrier<PIPE_V>();
                BeginOutput(bank);
                DataCopy(qg_bf16_gm_[output_offset], token_out_bf16, width_);
                DataCopy(
                    kg_bf16_gm_[output_offset], token_out_bf16[width_],
                    width_);
                DataCopy(
                    ag_bf16_gm_[output_offset], token_out_bf16[2 * width_],
                    width_);
                DataCopy(
                    bg_bf16_gm_[output_offset], token_out_bf16[3 * width_],
                    width_);
                DataCopy(
                    v_chunk_bf16_gm_[output_offset],
                    token_out_bf16[4 * width_], width_);
                DataCopy(
                    q_state_bf16_gm_[output_offset],
                    token_out_bf16[5 * width_], width_);
                DataCopy(
                    state_keys_bf16_gm_[output_offset],
                    token_out_bf16[6 * width_], width_);
                DataCopy(
                    state_keys_bf16_gm_[(chunk_size_ + token) * width_],
                    token_out_bf16[7 * width_], width_);
                EndOutput(bank);
                continue;
            }
            BeginOutput(bank);
            DataCopy(qg_gm_[output_offset], token_out, width_);
            DataCopy(kg_gm_[output_offset], token_out[width_], width_);
            DataCopy(ag_gm_[output_offset], token_out[2 * width_], width_);
            DataCopy(bg_gm_[output_offset], token_out[3 * width_], width_);
            DataCopy(
                v_chunk_gm_[output_offset], token_out[4 * width_], width_);
            DataCopy(
                q_state_gm_[output_offset], token_out[5 * width_], width_);
            DataCopy(
                state_keys_gm_[output_offset], token_out[6 * width_],
                width_);
            DataCopy(
                state_keys_gm_[(chunk_size_ + token) * width_],
                token_out[7 * width_], width_);
            EndOutput(bank);
        }
        FlushInputs();
        for (uint32_t bank = 0; bank < 2 && bank < chunk_size_; ++bank) {
            WaitOutput(bank);
        }

        for (uint32_t bank = 0; bank < 2; ++bank) {
            pipe_.ReleaseEventID<HardEvent::MTE2_V>(input_ready_[bank]);
            pipe_.ReleaseEventID<HardEvent::V_MTE2>(input_done_[bank]);
        }
        for (uint32_t bank = 0; bank < 2; ++bank) {
            pipe_.ReleaseEventID<HardEvent::V_MTE3>(output_ready_[bank]);
            pipe_.ReleaseEventID<HardEvent::MTE3_V>(output_done_[bank]);
        }
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> input_buffer_, float_buffer_, output_buffer_;
    TBuf<TPosition::VECCALC> inclusive_buffer_, output_bf16_buffer_;
    GlobalTensor<half> log_decay_gm_, k_gm_, v_gm_, kk_gm_, a_gm_, r_gm_;
    GlobalTensor<float> qg_gm_, kg_gm_, ag_gm_, bg_gm_, v_chunk_gm_;
    GlobalTensor<float> q_state_gm_, state_keys_gm_;
    GlobalTensor<bfloat16_t> qg_bf16_gm_, kg_bf16_gm_, ag_bf16_gm_;
    GlobalTensor<bfloat16_t> bg_bf16_gm_, v_chunk_bf16_gm_;
    GlobalTensor<bfloat16_t> q_state_bf16_gm_, state_keys_bf16_gm_;
    GlobalTensor<float> end_decay_gm_, offset_exp_gm_;
    uint32_t tokens_, heads_, width_, chunk_size_, chunks_;
    uint32_t batch_, chunk_, head_, group_;
    bool output_bf16_;
    event_t input_ready_[2], input_done_[2];
    event_t output_ready_[2], output_done_[2];
};

extern "C" __global__ __aicore__ void rwkv_chunk_gate_prep_direct(
    GM_ADDR log_decay, GM_ADDR k, GM_ADDR v, GM_ADDR kk, GM_ADDR a,
    GM_ADDR r, GM_ADDR qg, GM_ADDR kg, GM_ADDR ag, GM_ADDR bg,
    GM_ADDR v_chunk, GM_ADDR q_state, GM_ADDR state_keys,
    GM_ADDR end_decay, GM_ADDR offset_exp, uint32_t tokens,
    uint32_t heads, uint32_t width, uint32_t chunk_size,
    uint32_t chunks, uint32_t output_bf16) {
    RwkvChunkGatePrepDirectKernel kernel;
    kernel.Init(
        log_decay, k, v, kk, a, r, qg, kg, ag, bg, v_chunk, q_state,
        state_keys, end_decay, offset_exp, tokens, heads, width,
        chunk_size, chunks, output_bf16);
    kernel.Process();
}
