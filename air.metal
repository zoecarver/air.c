// air.metal - kernels. S = fp32 scratch, M = fp32 smalls, W = 4-bit weights; offsets are in floats (S, M) or bytes (W).
#include <metal_stdlib>
using namespace metal;

#define NE 2304
#define HD 128
#define NH 24
#define NKV 8
#define KVD 1024
#define QD 3072
#define NREL 16
#define REL_EXT 1024
#define HIST 16
#define CONVW (KVD + KVD + NE + NE)
#define NFF 1152
#define NEXP 128
#define NUSED 8
#define SITU_BETA 4.0f
#define SITU_LB 25.0f

inline float tg_sum(float v, threadgroup float *red, uint tid, uint ntg) {
    v = simd_sum(v);
    uint sg = tid / 32, nsg = (ntg + 31) / 32;
    if (tid % 32 == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0; for (uint i = 0; i < nsg; i++) t += red[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t;
}

// ---------------- generic copy ----------------
struct PCopy { uint src, dst, n, nvec, sstride, dstride; };
kernel void k_copy(device float *S [[buffer(0)]], constant PCopy &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n * p.nvec) return;
    uint v = gid / p.n, i = gid % p.n;
    S[p.dst + v * p.dstride + i] = S[p.src + v * p.sstride + i];
}

// ---------------- rmsnorm (optional weight) ----------------
struct PNorm { uint in, out, w, has_w, n, instride, outstride; float eps; };
kernel void k_rmsnorm(device float *S [[buffer(0)]], device const float *M [[buffer(1)]], constant PNorm &p [[buffer(2)]],
                      uint v [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
    threadgroup float red[32];
    device float *x = S + p.in + v * p.instride; device float *y = S + p.out + v * p.outstride;
    float ss = 0; for (uint i = tid; i < p.n; i += ntg) ss += x[i] * x[i];
    ss = tg_sum(ss, red, tid, ntg);
    float r = rsqrt(ss / p.n + p.eps);
    for (uint i = tid; i < p.n; i += ntg) y[i] = x[i] * r * (p.has_w ? M[p.w + i] : 1.0f);
}

// ---------------- depth mix: out[q] = sum_s softmax(S_s . Q_q / sqrt(NE)) S_s ; blocks pre-normalized ----------------
struct PMix { uint nb, blk, blkstride, cur, curstride, q, nq, out, outstride, outqstride, nw0, nw1, nw2, has_nw; };
kernel void k_mix(device float *S [[buffer(0)]], device const float *M [[buffer(1)]], constant PMix &p [[buffer(2)]],
                  uint gid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
    threadgroup float red[32]; threadgroup float sc[24];
    uint b = gid / p.nq, qi = gid % p.nq;
    device const float *cur = S + p.cur + b * p.curstride; device const float *Q = M + p.q + qi * NE;
    float ss = 0; for (uint i = tid; i < NE; i += ntg) ss += cur[i] * cur[i];
    ss = tg_sum(ss, red, tid, ntg); float rc = rsqrt(ss / NE + 1e-6f);
    uint ns = p.nb + 1;
    for (uint s = 0; s < ns; s++) {
        float d = 0;
        if (s < p.nb) { device const float *st = S + p.blk + b * p.blkstride + s * NE; for (uint i = tid; i < NE; i += ntg) d += st[i] * Q[i]; }
        else for (uint i = tid; i < NE; i += ntg) d += cur[i] * rc * Q[i];
        d = tg_sum(d, red, tid, ntg);
        if (tid == 0) sc[s] = d * rsqrt((float)NE);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx = -INFINITY; for (uint s = 0; s < ns; s++) mx = max(mx, sc[s]);
    float z = 0; float w[24]; for (uint s = 0; s < ns; s++) { w[s] = exp(sc[s] - mx); z += w[s]; }
    device float *o = S + p.out + b * p.outstride + qi * p.outqstride;
    float s2 = 0;
    for (uint i = tid; i < NE; i += ntg) {
        float a = 0;
        for (uint s = 0; s < p.nb; s++) a += w[s] * S[p.blk + b * p.blkstride + s * NE + i];
        a += w[p.nb] * cur[i] * rc;
        a /= z; o[i] = a; s2 += a * a;
    }
    if (p.has_nw) {
        s2 = tg_sum(s2, red, tid, ntg); float r2 = rsqrt(s2 / NE + 1e-6f);
        uint nw = qi == 0 ? p.nw0 : qi == 1 ? p.nw1 : p.nw2;
        for (uint i = tid; i < NE; i += ntg) o[i] = o[i] * r2 * M[nw + i];
    }
}

// ---------------- loop controllers ----------------
// items (b, j): cur = S[cur + b*cbs + j*cjs], msg = S[msg + b*mbs + j*mjs]; h2/g/f item-major
struct PCtl { uint cur, cbs, cjs, msg, mbs, mjs, h2, g, f, J; };
kernel void k_ctl_prep(device float *S [[buffer(0)]], constant PCtl &p [[buffer(1)]],
                       uint it [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
    threadgroup float red[32];
    uint b = it / p.J, j = it % p.J;
    device const float *c = S + p.cur + b * p.cbs + j * p.cjs, *m = S + p.msg + b * p.mbs + j * p.mjs;
    float s1 = 0, s2 = 0; for (uint i = tid; i < NE; i += ntg) { s1 += c[i] * c[i]; s2 += m[i] * m[i]; }
    s1 = tg_sum(s1, red, tid, ntg); s2 = tg_sum(s2, red, tid, ntg);
    float r1 = rsqrt(s1 / NE + 1e-6f), r2 = rsqrt(s2 / NE + 1e-6f);
    device float *h = S + p.h2 + it * 2 * NE;
    for (uint i = tid; i < NE; i += ntg) { h[i] = c[i] * r1; h[NE + i] = m[i] * r2; }
}
kernel void k_ctl_combine(device float *S [[buffer(0)]], constant PCtl &p [[buffer(1)]], uint gid [[thread_position_in_grid]], uint nitems [[threads_per_grid]]) {
    uint it = gid / NE, i = gid % NE; (void)nitems;
    uint b = it / p.J, j = it % p.J;
    device float *c = S + p.cur + b * p.cbs + j * p.cjs; device const float *m = S + p.msg + b * p.mbs + j * p.mjs;
    float g = S[p.g + it * NE + i], f = S[p.f + it * NE + i];
    c[i] = (1.0f - tanh(f)) * c[i] + tanh(g) * m[i];
}

// ---------------- small fp32 GEMV: y[b][n] = W[n] . x[b] ----------------
struct PGemv { uint w, N, K, x, xstride, y, ystride, B; };
kernel void k_gemv(device float *S [[buffer(0)]], device const float *M [[buffer(1)]], constant PGemv &p [[buffer(2)]],
                   uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint n = tg * 8 + sg; if (n >= p.N) return;
    device const float4 *w = (device const float4 *)(M + p.w + (size_t)n * p.K);
    float a[16]; _Pragma("unroll") for (uint b = 0; b < 16; b++) a[b] = 0;
    for (uint k = lane; k < p.K / 4; k += 32) { float4 wv = w[k];
        _Pragma("unroll") for (uint b = 0; b < 16; b++) if (b < p.B) a[b] += dot(wv, ((device const float4 *)(S + p.x + b * p.xstride))[k]); }
    _Pragma("unroll") for (uint b = 0; b < 16; b++) if (b < p.B) { float v = simd_sum(a[b]); if (lane == 0) S[p.y + b * p.ystride + n] = v; }
}

// ---------------- conv ring write: ring[(pos0+b) % HIST][off + c] = src[b][c] ----------------
struct PRing { uint src, sstride, off, nch, pos0, B; };
kernel void k_ring_write(device const float *S [[buffer(0)]], device float *R [[buffer(1)]], constant PRing &p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= p.B * p.nch) return;
    uint b = gid / p.nch, c = gid % p.nch;
    R[((p.pos0 + b) % HIST) * CONVW + p.off + c] = S[p.src + b * p.sstride + c];
}
// x(pp) = in-batch positions (pp >= pos0) from the source rows, older ones from the ring
inline float conv_at2(device const float *R, device const float *src, uint sstride, uint pos0, uint pos, uint off, uint c,
                      device const float *w, float scale, device const float *dyn, uint hw) {
    float acc = 0, xt = 0;
    for (uint j = 0; j < 4; j++) {
        int pp = (int)pos - 3 + (int)j; if (pp < 0) continue;
        float x = pp >= (int)pos0 ? src[(pp - pos0) * sstride + c] : R[(pp % HIST) * CONVW + off + c];
        acc += w[c * 4 + j] * x; if (j == 3) xt = x;
        if (dyn) acc += x * dyn[(3 - j) + 4 * (c / hw)];
    }
    return (xt + acc) * scale;
}

// ---------------- K/V conv + clip/norm + store (one threadgroup of 128 per (b, kvh, kind)) ----------------
struct PConvKV { uint pos0, B, cap, ring, offk, offv, wk, wv, dynk, dynv, has_dyn; float qk_clip, eps; uint srck, srcv, has8, ntile, vsoff, noring; };
inline uint v8_off(uint kh, uint ntile, uint pos, uint d) { return ((kh * ntile + pos / 32) * HD + d) * 32 + pos % 32; }
inline float tg_max(float v, threadgroup float *red, uint tid, uint ntg) {
    v = simd_max(v); uint sg = tid / 32, nsg = (ntg + 31) / 32;
    if (tid % 32 == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0; for (uint i = 0; i < nsg; i++) t = max(t, red[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t;
}
kernel void k_conv_kv(device float *S [[buffer(0)]], device const float *M [[buffer(1)]], device float *R [[buffer(2)]],
                      device half *Kc [[buffer(3)]], device half *Vc [[buffer(4)]], constant PConvKV &p [[buffer(5)]],
                      device char *K8 [[buffer(6)]], device char *V8 [[buffer(7)]], device float *KS [[buffer(8)]],
                      uint gid [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
    threadgroup float red[32];
    uint kind = gid % 2, kh = (gid / 2) % NKV, b = gid / (2 * NKV);
    uint pos = p.pos0 + b, c = kh * HD + d;
    float scale = rsqrt((p.has_dyn ? 2.0f : 1.0f) * 4.0f + 1.0f);
    device const float *dyn = p.has_dyn ? S + (kind ? p.dynv : p.dynk) + b * 32 : nullptr;
    device const float *src = S + (kind ? p.srcv : p.srck);
    float v = conv_at2(R, src, KVD, p.pos0, pos, kind ? p.offv : p.offk, c, M + (kind ? p.wv : p.wk), scale, dyn, HD);
    float ss = tg_sum(v * v, red, d, HD);
    if (kind == 0) { float nrm = sqrt(ss) + 1e-6f; float f = clamp(p.qk_clip / nrm, 0.0f, 1.0f); v *= f; }
    else v *= rsqrt(ss / HD + p.eps);
    uint slot = p.ring ? pos % p.cap : pos;
    if (kind == 0) Kc[(size_t)slot * KVD + c] = half(v); else Vc[(size_t)slot * KVD + c] = half(v);
    if (p.has8) { float a = tg_max(fabs(v), red, d, HD), sd = a / 127.0f, iv = sd > 0 ? 1.0f / sd : 0.0f; char q = char(rint(v * iv));
        if (kind == 0) { K8[((size_t)kh * p.ntile * 32 + pos) * HD + d] = q; if (d == 0) KS[pos * NKV + kh] = sd; }
        else { V8[v8_off(kh, p.ntile, pos, d)] = q; if (d == 0) KS[p.vsoff + pos * NKV + kh] = sd; } }
    // ring write of this position's raw input (readers of in-batch positions use the source rows)
    if (!p.noring) R[(pos % HIST) * CONVW + (kind ? p.offv : p.offk) + c] = src[b * KVD + c];
}

// ---------------- q / r clip + log-scale (one threadgroup of 128 per (b, head)) ----------------
struct PQR { uint q, r, qclip, rclip, full; float qk_clip, rel_clip; uint pos0; float alpha, floor_; };
kernel void k_qr(device float *S [[buffer(0)]], device const float *M [[buffer(1)]], constant PQR &p [[buffer(2)]],
                 uint gid [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
    threadgroup float red[32];
    uint b = gid / NH, h = gid % NH;
    device float *q = S + p.q + (b * NH + h) * HD;
    float v = q[d] * M[p.qclip + h * HD + d];
    float ss = tg_sum(v * v, red, d, HD);
    float f = clamp(p.qk_clip / (sqrt(ss) + 1e-6f), 0.0f, 1.0f);
    float tau = 1.0f + p.alpha * log(max(1.0f, (float)(p.pos0 + b + 1) / p.floor_));
    v *= f; if (p.full) v *= tau;
    q[d] = v;
    float rv = 0; if (d < NREL) rv = S[p.r + (b * NH + h) * NREL + d] * M[p.rclip + h * NREL + d];
    float rs = tg_sum(rv * rv, red, d, HD);
    if (d < NREL) { float fr = clamp(p.rel_clip / (sqrt(rs) + 1e-6f), 0.0f, 1.0f); rv *= fr; if (p.full) rv *= tau;
        S[p.r + (b * NH + h) * NREL + d] = rv; }
}

// merge partials + sink, multiply by sigmoid(gate): one threadgroup of 128 per (b, head)
struct PMerge { uint nchunk, pm, pl, pacc, sink, gate, att, has_sink, has_gate; };
// flash-decode attention: one threadgroup (256) per (kv head, chunk of CHK keys), online softmax over 256-key sub-blocks.
// scores: one key per thread (32 independent half4 loads -> high memory-level parallelism), queries broadcast from tg memory.
// values: thread = (dim, key half); V read once per chunk.
#define MAXQ 24
#define CHK 512
#define MQ2 12
struct PAttn { uint nq, pos0, k0, kend, cap, ring, swa, nchunk, q, r, relb, pm, pl, pacc, causal, has_bias; };
kernel void k_attn(device const half *K [[buffer(0)]], device const half *V [[buffer(1)]], device float *S [[buffer(2)]],
                    device const float *M [[buffer(3)]], constant PAttn &p [[buffer(4)]],
                    uint tid [[thread_position_in_threadgroup]], uint gid [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint kh = gid / p.nchunk, ch = gid % p.nchunk;
    const int chk = p.swa ? 256 : CHK;
    int c0 = p.k0 + ch * chk, c1 = min(c0 + chk - 1, (int)p.kend);
    threadgroup half4 qs[MQ2][HD / 4]; threadgroup float sc[MQ2][256]; threadgroup float rs[MQ2][NREL];
    threadgroup float mrun[MQ2], lrun[MQ2], alpha[MQ2]; threadgroup half4 ph[MQ2][64];
    const float isq = rsqrt((float)HD);
    for (uint i = tid; i < p.nq * HD; i += 256) { uint j = i / HD, d = i % HD; uint b = j / 3, h = kh * 3 + j % 3;
        ((threadgroup half *)qs[j])[d] = half(S[p.q + (b * NH + h) * HD + d] * isq); }
    if (p.has_bias) for (uint i = tid; i < p.nq * NREL; i += 256) { uint j = i / NREL, r = i % NREL; uint b = j / 3, h = kh * 3 + j % 3;
        rs[j][r] = S[p.r + (b * NH + h) * NREL + r]; }
    if (tid < MQ2) { mrun[tid] = -INFINITY; lrun[tid] = 0; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint d = tid % HD, kh2 = tid / HD;
    float acc[MQ2]; _Pragma("unroll") for (uint j = 0; j < MQ2; j++) acc[j] = 0;
    for (int s0 = c0; s0 <= c1; s0 += 256) {
        /* scores: thread = key */
        { int pp = s0 + (int)tid;
          if (pp <= c1) {
            uint sp = p.ring ? pp % p.cap : pp;
            device const half4 *kr = (device const half4 *)(K + (size_t)sp * KVD + kh * HD);
            half4 kv[32]; _Pragma("unroll") for (uint i = 0; i < 32; i++) kv[i] = kr[i];
            _Pragma("unroll") for (uint j = 0; j < MQ2; j++) if (j < p.nq) {
                uint b = j / 3; int dist = p.causal ? (int)(p.pos0 + b) - pp : (int)p.pos0 - pp;
                float s = 0; _Pragma("unroll") for (uint i = 0; i < 32; i++) s += float(dot(kv[i], qs[j][i]));
                if (p.has_bias && dist >= 0 && dist < REL_EXT) { device const float *rb = M + p.relb + dist * NREL;
                    _Pragma("unroll") for (uint r = 0; r < NREL; r++) s += rs[j][r] * rb[r]; }
                sc[j][tid] = ((p.causal && dist < 0) || (p.swa && dist >= 1024)) ? -INFINITY : s; }
          } else { _Pragma("unroll") for (uint j = 0; j < MQ2; j++) if (j < p.nq) sc[j][tid] = -INFINITY; } }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        /* online softmax update per query (simdgroup j % 8, lanes over keys, 8 keys per lane) */
        for (uint j = sg; j < p.nq; j += 8) {
            float4 v0 = float4(sc[j][lane * 8], sc[j][lane * 8 + 1], sc[j][lane * 8 + 2], sc[j][lane * 8 + 3]);
            float4 v1 = float4(sc[j][lane * 8 + 4], sc[j][lane * 8 + 5], sc[j][lane * 8 + 6], sc[j][lane * 8 + 7]);
            float mx = simd_max(max(max(max(v0.x, v0.y), max(v0.z, v0.w)), max(max(v1.x, v1.y), max(v1.z, v1.w))));
            float mo = mrun[j], mn = max(mo, mx);
            float a = (mn == -INFINITY) ? 1.0f : exp(mo - mn);
            float4 e0 = (mn == -INFINITY) ? float4(0) : exp(v0 - mn), e1 = (mn == -INFINITY) ? float4(0) : exp(v1 - mn);
            ph[j][lane * 2] = half4(e0); ph[j][lane * 2 + 1] = half4(e1);
            float l = simd_sum(e0.x + e0.y + e0.z + e0.w + e1.x + e1.y + e1.z + e1.w);
            if (lane == 0) { alpha[j] = a; lrun[j] = lrun[j] * a + l; mrun[j] = mn; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        /* values */
        _Pragma("unroll") for (uint j = 0; j < MQ2; j++) if (j < p.nq) acc[j] *= alpha[j];
        { int t0 = kh2 * 128, t1 = min(t0 + 128, c1 - s0 + 1);       /* probabilities beyond t1 are 0: clamp the loads only */
          for (int tt = t0; tt < t0 + 128; tt += 32) { if (tt >= t1) break;
            half4 vh[8];
            _Pragma("unroll") for (uint u = 0; u < 8; u++) { half4 v;
                _Pragma("unroll") for (uint w = 0; w < 4; w++) { int t = tt + u * 4 + w; int pp = s0 + min(t, t1 - 1); uint sp = p.ring ? pp % p.cap : pp;
                    v[w] = V[(size_t)sp * KVD + kh * HD + d]; }
                vh[u] = v; }
            _Pragma("unroll") for (uint j = 0; j < MQ2; j++) if (j < p.nq) {
                threadgroup const half4 *pj = &ph[j][tt / 4]; half h = 0;
                _Pragma("unroll") for (uint u = 0; u < 8; u++) h += dot(pj[u], vh[u]);
                acc[j] += float(h); } } }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (kh2 == 1) { _Pragma("unroll") for (uint j = 0; j < MQ2; j++) if (j < p.nq) sc[j][d] = acc[j]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kh2 == 0) { _Pragma("unroll") for (uint j = 0; j < MQ2; j++) if (j < p.nq) S[p.pacc + ((size_t)gid * MAXQ + j) * HD + d] = acc[j] + sc[j][d]; }
    if (tid < p.nq) { S[p.pm + gid * MAXQ + tid] = mrun[tid]; S[p.pl + gid * MAXQ + tid] = lrun[tid]; }
}
struct PAttn8 { uint nq, pos0, k0, kend, nchunk, q, r, relb, pm, pl, pacc, has_bias, ntile, vsoff; };
kernel void k_attn8(device const char *K8 [[buffer(0)]], device const char *V8 [[buffer(1)]], device float *S [[buffer(2)]],
                    device const float *M [[buffer(3)]], constant PAttn8 &p [[buffer(4)]], device const float *KS [[buffer(5)]],
                    uint tid [[thread_position_in_threadgroup]], uint gid [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint kh = gid / p.nchunk, ch = gid % p.nchunk;
    int c0 = p.k0 + ch * CHK, c1 = min(c0 + CHK - 1, (int)p.kend);
    threadgroup half4 qs[MQ2][HD / 4]; threadgroup float sc[MQ2][256]; threadgroup float rs[MQ2][NREL];
    threadgroup float mrun[MQ2], lrun[MQ2], alpha[MQ2]; threadgroup half4 ph[MQ2][64];
    const float isq = rsqrt((float)HD);
    for (uint i = tid; i < p.nq * HD; i += 256) { uint j = i / HD, d = i % HD; uint b = j / 3, h = kh * 3 + j % 3;
        ((threadgroup half *)qs[j])[d] = half(S[p.q + (b * NH + h) * HD + d] * isq); }
    if (p.has_bias) for (uint i = tid; i < p.nq * NREL; i += 256) { uint j = i / NREL, r = i % NREL; uint b = j / 3, h = kh * 3 + j % 3;
        rs[j][r] = S[p.r + (b * NH + h) * NREL + r]; }
    if (tid < MQ2) { mrun[tid] = -INFINITY; lrun[tid] = 0; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint d = tid % HD, kh2 = tid / HD;
    float acc[MQ2]; _Pragma("unroll") for (uint j = 0; j < MQ2; j++) acc[j] = 0;
    for (int s0 = c0; s0 <= c1; s0 += 256) {
        { int pp = s0 + (int)tid;
          if (pp <= c1) {
            device const uint4 *kr = (device const uint4 *)(K8 + ((size_t)kh * p.ntile * 32 + pp) * HD);
            uint4 raw[8]; _Pragma("unroll") for (uint i = 0; i < 8; i++) raw[i] = kr[i];
            half4 kv[32]; _Pragma("unroll") for (uint i = 0; i < 8; i++) { _Pragma("unroll") for (uint c = 0; c < 4; c++) kv[i * 4 + c] = half4(as_type<char4>(raw[i][c])); }
            float ks = KS[pp * NKV + kh];
            _Pragma("unroll") for (uint j = 0; j < MQ2; j++) if (j < p.nq) {
                uint b = j / 3; int dist = (int)(p.pos0 + b) - pp;
                half hs0 = 0, hs1 = 0; _Pragma("unroll") for (uint i = 0; i < 16; i++) { hs0 += dot(kv[i], qs[j][i]); hs1 += dot(kv[16 + i], qs[j][16 + i]); }
                float s = (float(hs0) + float(hs1)) * ks;
                if (p.has_bias && dist >= 0 && dist < REL_EXT) { device const float *rb = M + p.relb + dist * NREL;
                    _Pragma("unroll") for (uint r = 0; r < NREL; r++) s += rs[j][r] * rb[r]; }
                sc[j][tid] = dist < 0 ? -INFINITY : s; }
          } else { _Pragma("unroll") for (uint j = 0; j < MQ2; j++) if (j < p.nq) sc[j][tid] = -INFINITY; } }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = sg; j < p.nq; j += 8) {
            float vsr[8]; for (uint u = 0; u < 8; u++) { int pp = s0 + lane * 8 + u; vsr[u] = pp <= c1 ? KS[p.vsoff + pp * NKV + kh] : 0.0f; }
            float4 v0 = float4(sc[j][lane * 8], sc[j][lane * 8 + 1], sc[j][lane * 8 + 2], sc[j][lane * 8 + 3]);
            float4 v1 = float4(sc[j][lane * 8 + 4], sc[j][lane * 8 + 5], sc[j][lane * 8 + 6], sc[j][lane * 8 + 7]);
            float mx = simd_max(max(max(max(v0.x, v0.y), max(v0.z, v0.w)), max(max(v1.x, v1.y), max(v1.z, v1.w))));
            float mo = mrun[j], mn = max(mo, mx);
            float a = (mn == -INFINITY) ? 1.0f : exp(mo - mn);
            float4 e0 = (mn == -INFINITY) ? float4(0) : exp(v0 - mn), e1 = (mn == -INFINITY) ? float4(0) : exp(v1 - mn);
            float l = simd_sum(e0.x + e0.y + e0.z + e0.w + e1.x + e1.y + e1.z + e1.w);
            ph[j][lane * 2] = half4(e0 * float4(vsr[0], vsr[1], vsr[2], vsr[3])); ph[j][lane * 2 + 1] = half4(e1 * float4(vsr[4], vsr[5], vsr[6], vsr[7]));
            if (lane == 0) { alpha[j] = a; lrun[j] = lrun[j] * a + l; mrun[j] = mn; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        _Pragma("unroll") for (uint j = 0; j < MQ2; j++) if (j < p.nq) acc[j] *= alpha[j];
        for (uint tt = 0; tt < 4; tt++) {          /* 4 tiles of 32 positions per key half */
            uint t0 = kh2 * 128 + tt * 32; int pbase = s0 + (int)t0; if (pbase > c1) break;
            device const uint4 *vt = (device const uint4 *)(V8 + v8_off(kh, p.ntile, pbase, d));
            uint4 r0 = vt[0], r1 = vt[1]; uint vw[8] = { r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w };
            half4 vh[8]; _Pragma("unroll") for (uint u = 0; u < 8; u++) vh[u] = half4(as_type<char4>(vw[u]));
            _Pragma("unroll") for (uint j = 0; j < MQ2; j++) if (j < p.nq) {
                threadgroup const half4 *pj = &ph[j][t0 / 4]; half h = 0;
                _Pragma("unroll") for (uint u = 0; u < 8; u++) h += dot(pj[u], vh[u]);
                acc[j] += float(h); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (kh2 == 1) { _Pragma("unroll") for (uint j = 0; j < MQ2; j++) if (j < p.nq) sc[j][d] = acc[j]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kh2 == 0) { _Pragma("unroll") for (uint j = 0; j < MQ2; j++) if (j < p.nq) S[p.pacc + ((size_t)gid * MAXQ + j) * HD + d] = acc[j] + sc[j][d]; }
    if (tid < p.nq) { S[p.pm + gid * MAXQ + tid] = mrun[tid]; S[p.pl + gid * MAXQ + tid] = lrun[tid]; }
}
kernel void k_attn_merge(device float *S [[buffer(0)]], constant PMerge &p [[buffer(1)]],
                         uint gid [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
    uint b = gid / NH, h = gid % NH, kh = h / 3, j = b * 3 + h % 3;
    float t = p.has_sink ? S[p.sink + b * NH + h] : -INFINITY, M = t;
    for (uint c = 0; c < p.nchunk; c++) M = max(M, S[p.pm + (kh * p.nchunk + c) * MAXQ + j]);
    float z = p.has_sink ? exp(t - M) : 0.0f, o = 0;
    for (uint c = 0; c < p.nchunk; c++) { uint it = kh * p.nchunk + c; float mm = S[p.pm + it * MAXQ + j]; if (mm == -INFINITY) continue;
        float f = exp(mm - M); z += f * S[p.pl + it * MAXQ + j]; o += f * S[p.pacc + ((size_t)it * MAXQ + j) * HD + d]; }
    float r = o / z;
    if (p.has_gate) r /= 1.0f + exp(-S[p.gate + b * QD + h * HD + d]);
    S[p.att + b * QD + h * HD + d] = r;
}

// ---------------- ffn post conv + residual: cur = ffn_in + conv(src) (threadgroup per position) ----------------
struct PPost { uint pos0, off, w, cur, ffn, src, noring; };
kernel void k_post(device float *S [[buffer(0)]], device const float *M [[buffer(1)]], device float *R [[buffer(2)]], constant PPost &p [[buffer(3)]],
                   uint b [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
    for (uint i = tid; i < NE; i += ntg) {
        float oc = conv_at2(R, S + p.src, NE, p.pos0, p.pos0 + b, p.off, i, M + p.w, rsqrt(5.0f), nullptr, HD);
        if (!p.noring) R[((p.pos0 + b) % HIST) * CONVW + p.off + i] = S[p.src + b * NE + i];
        S[p.cur + b * NE + i] = S[p.ffn + b * NE + i] + oc;
    }
}

// ---------------- MoE routing ----------------

// parallel top-8 over 128 experts with 128 threads; ties -> lower index
inline void topk8(threadgroup float *sel, threadgroup float *prob, uint e, device uint *ids_out, device float *wts_out, float wscale,
                  threadgroup float *rv, threadgroup uint *ri, threadgroup uint *chosen) {
    for (uint k = 0; k < NUSED; k++) {
        float v = sel[e]; uint idx = e;
        for (uint off = 16; off > 0; off >>= 1) { float ov = simd_shuffle_down(v, off); uint oi = simd_shuffle_down(idx, off);
            if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; } }
        if (e % 32 == 0) { rv[e / 32] = v; ri[e / 32] = idx; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (e == 0) { float bv = rv[0]; uint bi = ri[0]; for (uint s = 1; s < 4; s++) if (rv[s] > bv || (rv[s] == bv && ri[s] < bi)) { bv = rv[s]; bi = ri[s]; }
            chosen[k] = bi; sel[bi] = -INFINITY; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (e == 0) { float wsum = 0; for (uint k = 0; k < NUSED; k++) wsum += prob[chosen[k]]; wsum += 1e-20f;
        for (uint k = 0; k < NUSED; k++) { ids_out[k] = chosen[k]; wts_out[k] = prob[chosen[k]] / wsum * wscale; } }
}
struct PTopk { uint logit, bias, ids, wts; float wscale; };
kernel void k_topk(device float *S [[buffer(0)]], device const float *M [[buffer(1)]], constant PTopk &p [[buffer(2)]],
                   uint b [[threadgroup_position_in_grid]], uint e [[thread_index_in_threadgroup]]) {
    threadgroup float sel[NEXP], prob[NEXP], rv[4]; threadgroup uint ri[4], chosen[NUSED];
    float pr = 1.0f / (1.0f + exp(-S[p.logit + b * NEXP + e]));
    prob[e] = pr; sel[e] = pr + M[p.bias + e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    topk8(sel, prob, e, ((device uint *)(S + p.ids)) + b * NUSED, S + p.wts + b * NUSED, p.wscale, rv, ri, chosen);
}

struct Job { uint woff_lo, woff_hi, N, K; float gscale; uint B, x0, x1, x2, x3, yoff, ystride, tile0; char tbl[16]; uint e2m1, pad[2]; };
struct ETab { uint gu_lo, gu_hi, dn_lo, dn_hi; float gsg, gsd; uint e2m1; char tbl[16]; uint pad; };
// one thread builds the expert union and the job lists (gate_up and down, per B class 1..4)
struct PPlan { uint B, layer, ids, xm, gu, act, dn, slotmap, jobs, counts, args, maxjobs; };
kernel void k_moe_plan(device float *S [[buffer(0)]], device const ETab *ET [[buffer(1)]], device Job *J [[buffer(2)]],
                       device uint *CNT [[buffer(3)]], device uint *ARGS [[buffer(4)]], constant PPlan &p [[buffer(5)]], uint t [[thread_index_in_threadgroup]]) {
    threadgroup uint ex[128], first[128], uofs[128], ufirst[128], cnt[128], nun;
    device const uint *ids = ((device const uint *)(S + p.ids));
    device uint *slotmap = ((device uint *)(S + p.slotmap));
    const uint ns = p.B * NUSED;
    if (t < ns) ex[t] = ids[t];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < ns) { uint f = ex[t] < NEXP; for (uint j = 0; j < t && f; j++) if (ex[j] == ex[t]) { f = 0; break; } first[t] = f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < ns && first[t]) { uint uidx = 0; for (uint j = 0; j < t; j++) uidx += first[j]; uofs[uidx] = t; uint c = 0; for (uint j = t; j < ns; j++) c += ex[j] == ex[t]; cnt[uidx] = c; ufirst[t] = uidx; }
    if (t == 0) { uint n = 0; for (uint j = 0; j < ns; j++) n += first[j]; nun = n; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint un = nun, TG = (2 * NFF / 4 + 31) / 32, TD = (NE / 4 + 31) / 32;
    if (t < un) {
        uint s0 = uofs[t], e = ex[s0], rowb = 0, jobb = 0;
        for (uint v = 0; v < t; v++) { rowb += cnt[v]; jobb += (cnt[v] + 3) / 4; }
        device const ETab &et = ET[p.layer * NEXP + e];
        uint c = cnt[t], k = 0; uint lst[16];
        for (uint j = s0; j < ns && k < 16; j++) if (ex[j] == e) lst[k++] = j;
        for (uint c0 = 0; c0 < c; c0 += 4) {
            uint nb = min(4u, c - c0), row = rowb + c0, ji = jobb + c0 / 4;
            Job g; g.woff_lo = et.gu_lo; g.woff_hi = et.gu_hi; g.N = 2 * NFF; g.K = NE; g.gscale = et.gsg; g.B = nb; g.e2m1 = et.e2m1; g.pad[0] = 0; g.pad[1] = 0;
            for (uint i = 0; i < 16; i++) g.tbl[i] = et.tbl[i];
            Job d = g; d.woff_lo = et.dn_lo; d.woff_hi = et.dn_hi; d.N = NE; d.K = NFF; d.gscale = et.gsd; d.pad[0] = 1;
            uint xs[4] = {0, 0, 0, 0}, gs[4] = {0, 0, 0, 0};
            for (uint i = 0; i < nb; i++) { uint sl = lst[c0 + i]; xs[i] = p.xm + (sl / NUSED) * NE; gs[i] = p.gu + (row + i) * 2 * NFF; slotmap[sl] = row + i; }
            g.x0 = xs[0]; g.x1 = xs[1]; g.x2 = xs[2]; g.x3 = xs[3]; g.yoff = p.gu + row * 2 * NFF; g.ystride = 2 * NFF; g.tile0 = ji * TG;
            d.x0 = gs[0]; d.x1 = gs[1]; d.x2 = gs[2]; d.x3 = gs[3]; d.yoff = p.dn + row * NE; d.ystride = NE; d.tile0 = ji * TD;
            J[ji] = g; J[p.maxjobs + ji] = d;
        }
        if (t == un - 1) { uint nj = jobb + (c + 3) / 4, rows = rowb + c;
            CNT[0] = nj; CNT[1] = nj; CNT[8] = rows; ARGS[0] = nj * TG; ARGS[1] = 1; ARGS[2] = 1; ARGS[3] = nj * TD; ARGS[4] = 1; ARGS[5] = 1; }
    }
}
struct PComb { uint slotmap, wts, dn, out, B; };
kernel void k_moe_combine(device float *S [[buffer(0)]], constant PComb &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= p.B * NE) return;
    uint b = gid / NE, i = gid % NE; float a = 0;
    device const uint *sm = ((device const uint *)(S + p.slotmap));
    for (uint k = 0; k < NUSED; k++) { float w = S[p.wts + b * NUSED + k]; if (w != 0.0f) a += w * S[p.dn + sm[b * NUSED + k] * NE + i]; }
    S[p.out + b * NE + i] = a;
}

// ---------------- 4-bit NV4R matmul (job list; count from a device buffer) ----------------
inline half4 e2m1x4(uint w, uint sh) {
    ushort4 c = ushort4((w >> sh) & 15u, (w >> (sh + 8)) & 15u, (w >> (sh + 16)) & 15u, (w >> (sh + 24)) & 15u);
    ushort4 m = c & ushort4(7), s = (c & ushort4(8)) << 12;
    ushort4 b = select(select(ushort4(0), ushort4(0x3800), m == ushort4(1)), (m << 9) + ushort4(0x3800), m >= ushort4(2));
    return as_type<half4>(b | s);
}
template <uint FB> void mm4_body(device const uchar *W, device const float *X, device float *Y, device const Job &jb, uint tg,
  uint sg, uint lane, uint tid, threadgroup half2 *lut, threadgroup half *tx, threadgroup float *red) {
  /* byte -> (codebook[lo nibble], codebook[hi nibble]), |codebook| <= 12 */
  const float lsc = jb.e2m1 ? 1.0f : 0.0625f;
  { uint c = tid; lut[c] = half2(half(jb.tbl[c & 15] * lsc), half(jb.tbl[c >> 4] * lsc)); }
  const uint B = FB; uint KB = jb.K / 16; ulong woff = ((ulong)jb.woff_hi << 32) | jb.woff_lo;
  uint xo0[4] = { jb.x0, jb.x1, jb.x2, jb.x3 };
  float xs[4] = {1, 1, 1, 1};
  if (jb.pad[0]) {   /* SITU staging (|act| <= 100: no scaling needed) */
    for (uint i = tid; i < B * jb.K; i += 256) { uint p = i / jb.K, k = i % jb.K; float g = X[xo0[p] + k], up = X[xo0[p] + jb.K + k];
      float act = SITU_BETA * tanh(g / SITU_BETA) / (1.0f + exp(-g)) * (SITU_LB * tanh(up / SITU_LB)); tx[p * jb.K + k] = half(act); }
  } else {
    /* per-position power-of-two scale so that 16 half products (|w| <= 12) stay < 65504: max|x * s| <= 256 */
    float mx[4] = {0, 0, 0, 0};
    for (uint i = tid; i < B * jb.K; i += 256) { uint p = i / jb.K, k = i % jb.K; mx[p] = max(mx[p], fabs(X[xo0[p] + k])); }
    for (uint p = 0; p < B; p++) { float v = simd_max(mx[p]); if (lane == 0) red[sg * 4 + p] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = 0; p < B; p++) { float v = 0; for (uint s = 0; s < 8; s++) v = max(v, red[s * 4 + p]);
      xs[p] = v > 256.0f ? exp2(ceil(log2(v / 256.0f))) : 1.0f; }
    for (uint i = tid; i < B * jb.K; i += 256) { uint p = i / jb.K, k = i % jb.K; tx[p * jb.K + k] = half(X[xo0[p] + k] / xs[p]); }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint ngi = jb.pad[1] ? jb.pad[1] : 4;      /* row-groups per simdgroup (tile = 32*ngi rows) */
  for (uint gi = 0; gi < ngi; gi++) {
    uint g = (tg - jb.tile0) * 8 * ngi + gi * 8 + sg; if (g * 4 >= jb.N) break;
    device const uchar *base = W + woff + (ulong)g * KB * 36;
    float acc[4][FB];
    for (uint rr = 0; rr < 4; rr++) for (uint p = 0; p < FB; p++) acc[rr][p] = 0;
    for (uint kb = lane; kb < KB; kb += 32) {
      device const uchar *sb = base + (kb >> 2) * 144;
      device const uint4 *nw4 = (device const uint4 *)(sb + (kb & 3) * 32);
      uint4 NA = nw4[0], NB = nw4[1];
      uint s4 = *(device const uint *)(sb + 128 + (kb & 3) * 4);
      half4 xv[FB][4];
      for (uint p = 0; p < B; p++) { threadgroup const half4 *xp = (threadgroup const half4 *)(tx + p * jb.K + kb * 16);
        xv[p][0] = xp[0]; xv[p][1] = xp[1]; xv[p][2] = xp[2]; xv[p][3] = xp[3]; }
      for (uint rr = 0; rr < 4; rr++) {
        uint A = NA[rr], Bw = NB[rr];
        float sc = as_type<float>((((s4 >> (8 * rr)) & 255u) << 20) + (117u << 23));
        half2 a0 = lut[A & 255], a1 = lut[(A >> 8) & 255], a2 = lut[(A >> 16) & 255], a3 = lut[A >> 24];
        half2 b0 = lut[Bw & 255], b1 = lut[(Bw >> 8) & 255], b2 = lut[(Bw >> 16) & 255], b3 = lut[Bw >> 24];
        half4 w0 = half4(a0.x, a1.x, a2.x, a3.x), w1 = half4(a0.y, a1.y, a2.y, a3.y);
        half4 w2 = half4(b0.x, b1.x, b2.x, b3.x), w3 = half4(b0.y, b1.y, b2.y, b3.y);
        for (uint p = 0; p < B; p++) { half h = dot(w0, xv[p][0]) + dot(w1, xv[p][1]) + dot(w2, xv[p][2]) + dot(w3, xv[p][3]);
          acc[rr][p] = fma(sc, float(h), acc[rr][p]); }
      }
    }
    for (uint rr = 0; rr < 4; rr++) for (uint p = 0; p < FB; p++) { float v = simd_sum(acc[rr][p]);
      if (lane == 0) Y[jb.yoff + p * jb.ystride + 4 * g + rr] = v * (jb.gscale / lsc) * xs[p]; }
  }
}
template <uint FB> void mm4_body_big(device const uchar *W, device const float *X, device float *Y, device const Job &jb, uint tg,
  uint sg, uint lane, uint tid, threadgroup half2 *lut, threadgroup half *tx) {
  const float lsc = jb.e2m1 ? 1.0f : 0.0625f;     /* keep |codebook| <= 12 so half dot products cannot overflow */
  { uint c = tid; lut[c] = half2(half(jb.tbl[c & 15] * lsc), half(jb.tbl[c >> 4] * lsc)); }
  const uint B = FB; uint KB = jb.K / 16; ulong woff = ((ulong)jb.woff_hi << 32) | jb.woff_lo;
  uint xo0[4] = { jb.x0, jb.x1, jb.x2, jb.x3 };
  const uint KC = (14336 / FB) / 64 * 64;                  /* staged K per chunk */
  float acc[4][4][FB];
  for (uint gi = 0; gi < 4; gi++) for (uint rr = 0; rr < 4; rr++) for (uint p = 0; p < FB; p++) acc[gi][rr][p] = 0;
  for (uint c0 = 0; c0 < jb.K; c0 += KC) {
    uint kc = min(KC, jb.K - c0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < B * kc; i += 256) { uint p = i / kc, k = i % kc; tx[p * KC + k] = half(X[xo0[p] + c0 + k]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint gi = 0; gi < 4; gi++) {
      uint g = (tg - jb.tile0) * 32 + gi * 8 + sg; if (g * 4 >= jb.N) break;
      device const uchar *base = W + woff + (ulong)g * KB * 36;
      for (uint kb = c0 / 16 + lane; kb < (c0 + kc) / 16; kb += 32) {
        device const uchar *sb = base + (kb >> 2) * 144;
        device const uint4 *nw4 = (device const uint4 *)(sb + (kb & 3) * 32);
        uint4 NA = nw4[0], NB = nw4[1];
        uint s4 = *(device const uint *)(sb + 128 + (kb & 3) * 4);
        half4 xv[FB][4];
        for (uint p = 0; p < B; p++) { threadgroup const half4 *xp = (threadgroup const half4 *)(tx + p * KC + (kb * 16 - c0));
          xv[p][0] = xp[0]; xv[p][1] = xp[1]; xv[p][2] = xp[2]; xv[p][3] = xp[3]; }
        for (uint rr = 0; rr < 4; rr++) {
          uint A = NA[rr], Bw = NB[rr];
          float sc = as_type<float>((((s4 >> (8 * rr)) & 255u) << 20) + (117u << 23));
          half2 a0 = lut[A & 255], a1 = lut[(A >> 8) & 255], a2 = lut[(A >> 16) & 255], a3 = lut[A >> 24];
          half2 b0 = lut[Bw & 255], b1 = lut[(Bw >> 8) & 255], b2 = lut[(Bw >> 16) & 255], b3 = lut[Bw >> 24];
          half4 w0 = half4(a0.x, a1.x, a2.x, a3.x), w1 = half4(a0.y, a1.y, a2.y, a3.y);
          half4 w2 = half4(b0.x, b1.x, b2.x, b3.x), w3 = half4(b0.y, b1.y, b2.y, b3.y);
          for (uint p = 0; p < B; p++) acc[gi][rr][p] += sc * ((float(dot(w0, xv[p][0])) + float(dot(w1, xv[p][1]))) + (float(dot(w2, xv[p][2])) + float(dot(w3, xv[p][3]))));
        }
      }
    }
  }
  for (uint gi = 0; gi < 4; gi++) { uint g = (tg - jb.tile0) * 32 + gi * 8 + sg; if (g * 4 >= jb.N) break;
    for (uint rr = 0; rr < 4; rr++) for (uint p = 0; p < FB; p++) { float v = simd_sum(acc[gi][rr][p]);
      if (lane == 0) Y[jb.yoff + p * jb.ystride + 4 * g + rr] = v * (jb.gscale / lsc); } }
}
inline uint find_job(device const Job *jobs, uint njobs, uint tg) {
  uint lo = 0, hi = njobs - 1; while (lo < hi) { uint mid = (lo + hi + 1) / 2; if (jobs[mid].tile0 <= tg) lo = mid; else hi = mid - 1; }
  return lo;
}
template <uint FB> kernel void mm4(device const uchar *W [[buffer(0)]], device const float *X [[buffer(1)]], device float *Y [[buffer(2)]],
  device const Job *jobs [[buffer(3)]], device const uint *njobs_p [[buffer(4)]], threadgroup half *tx [[threadgroup(0)]],
  uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
  uint tid [[thread_index_in_threadgroup]]) {
  uint njobs = *njobs_p; if (njobs == 0) return;
  threadgroup half2 lut[256]; threadgroup float red[32];
  mm4_body<FB>(W, X, Y, jobs[find_job(jobs, njobs, tg)], tg, sg, lane, tid, lut, tx, red);
}
/* MoE: one dispatch, mixed position counts; uniform branch per threadgroup */
kernel void mm4_moe(device const uchar *W [[buffer(0)]], device const float *X [[buffer(1)]], device float *Y [[buffer(2)]],
  device const Job *jobs [[buffer(3)]], device const uint *njobs_p [[buffer(4)]], threadgroup half *tx [[threadgroup(0)]],
  uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
  uint tid [[thread_index_in_threadgroup]]) {
  uint njobs = *njobs_p; if (njobs == 0) return;
  threadgroup half2 lut[256]; threadgroup float red[32];
  device const Job &jb = jobs[find_job(jobs, njobs, tg)];
  switch (jb.B) { case 1: mm4_body<1>(W, X, Y, jb, tg, sg, lane, tid, lut, tx, red); break; case 2: mm4_body<2>(W, X, Y, jb, tg, sg, lane, tid, lut, tx, red); break;
                  case 3: mm4_body<3>(W, X, Y, jb, tg, sg, lane, tid, lut, tx, red); break; default: mm4_body<4>(W, X, Y, jb, tg, sg, lane, tid, lut, tx, red); }
}
template <uint FB> kernel void mm4big(device const uchar *W [[buffer(0)]], device const float *X [[buffer(1)]], device float *Y [[buffer(2)]],
  device const Job *jobs [[buffer(3)]], device const uint *njobs_p [[buffer(4)]],
  uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
  uint tid [[thread_index_in_threadgroup]]) {
  uint njobs = *njobs_p; if (njobs == 0) return;
  threadgroup half2 lut[256]; threadgroup half tx[14336];
  mm4_body_big<FB>(W, X, Y, jobs[find_job(jobs, njobs, tg)], tg, sg, lane, tid, lut, tx);
}
#define INSTB(b) template [[host_name("mm4big_" #b)]] kernel void mm4big<b>(device const uchar *, device const float *, device float *, device const Job *, device const uint *, uint, uint, uint, uint);
INSTB(1) INSTB(2) INSTB(3) INSTB(4)
#define INST(b) template [[host_name("mm4_" #b)]] kernel void mm4<b>(device const uchar *, device const float *, device float *, device const Job *, device const uint *, threadgroup half *, uint, uint, uint, uint);
INST(1) INST(2) INST(3) INST(4)

// ---------------- top-16 over the vocabulary: pass a (per chunk), pass b (merge) ----------------
struct PTop { uint lg, lgstride, n, nchunk, part, K; };
kernel void k_top_a(device float *S [[buffer(0)]], constant PTop &p [[buffer(1)]],
                    uint gid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
    uint b = gid / p.nchunk, ch = gid % p.nchunk;
    uint len = (p.n + p.nchunk - 1) / p.nchunk, lo = ch * len, hi = min(lo + len, p.n);
    device const float *l = S + p.lg + b * p.lgstride;
    threadgroup float bv[256]; threadgroup uint bi[256]; threadgroup uint taken[64];
    for (uint r = 0; r < p.K; r++) {
        float best = -INFINITY; uint bidx = 0xffffffff;
        for (uint i = lo + tid; i < hi; i += ntg) { bool skip = false; for (uint q = 0; q < r; q++) if (taken[q] == i) skip = true;
            if (!skip && (l[i] > best || (l[i] == best && i < bidx))) { best = l[i]; bidx = i; } }
        bv[tid] = best; bi[tid] = bidx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = ntg / 2; s > 0; s >>= 1) { if (tid < s && (bv[tid + s] > bv[tid] || (bv[tid + s] == bv[tid] && bi[tid + s] < bi[tid]))) { bv[tid] = bv[tid + s]; bi[tid] = bi[tid + s]; }
            threadgroup_barrier(mem_flags::mem_threadgroup); }
        if (tid == 0) { taken[r] = bi[0]; S[p.part + (gid * p.K + r) * 2] = bv[0]; ((device uint *)(S + p.part))[(gid * p.K + r) * 2 + 1] = bi[0]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
// merge of the per-chunk partial top-K lists: one threadgroup of 1024 per b, K rounds of a parallel argmax
kernel void k_top_b(device float *S [[buffer(0)]], device int *OUT [[buffer(1)]], constant PTop &p [[buffer(2)]],
                    uint b [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    uint n = p.nchunk * p.K; device const float *pv = S + p.part + b * n * 2;
    float v[4]; uint id[4];
    for (uint k = 0; k < 4; k++) { uint i = tid + k * 1024; if (i < n) { v[k] = pv[i * 2]; id[k] = as_type<uint>(pv[i * 2 + 1]); if (id[k] == 0xffffffffu) v[k] = -INFINITY; }
                                   else { v[k] = -INFINITY; id[k] = 0xffffffffu; } }
    threadgroup float rv[32]; threadgroup uint ri[32]; threadgroup uint win;
    for (uint r = 0; r < p.K; r++) {
        float bv = v[0]; uint bi = id[0];
        for (uint k = 1; k < 4; k++) if (v[k] > bv || (v[k] == bv && id[k] < bi)) { bv = v[k]; bi = id[k]; }
        for (uint off = 16; off > 0; off >>= 1) { float ov = simd_shuffle_down(bv, off); uint oi = simd_shuffle_down(bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; } }
        if (lane == 0) { rv[sg] = bv; ri[sg] = bi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) { bv = rv[lane]; bi = ri[lane];
            for (uint off = 16; off > 0; off >>= 1) { float ov = simd_shuffle_down(bv, off); uint oi = simd_shuffle_down(bi, off);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; } }
            if (lane == 0) { win = bi; OUT[b * p.K + r] = (int)bi; } }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint w = win; for (uint k = 0; k < 4; k++) if (id[k] == w) v[k] = -INFINITY;
    }
}

struct PFront { uint pos0, off, w, cur, ffn, blk, blkstride, nb, push, src, frq, xm, route, nw, noring; };
kernel void k_ffn_front(device float *S [[buffer(0)]], device const float *M [[buffer(1)]], device float *R [[buffer(2)]], constant PFront &p [[buffer(3)]],
                        uint b [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
    threadgroup float red[32]; threadgroup float sc[24]; threadgroup float lg[NEXP];
    device float *cur = S + p.cur + b * NE;
    float rc = 1;
    if (p.push) { float ss = 0; for (uint i = tid; i < NE; i += ntg) ss += cur[i] * cur[i]; ss = tg_sum(ss, red, tid, ntg); rc = rsqrt(ss / NE + 1e-6f); }
    const float scale = rsqrt(5.0f);
    for (uint i = tid; i < NE; i += ntg) {
        float oc = conv_at2(R, S + p.src, NE, p.pos0, p.pos0 + b, p.off, i, M + p.w, scale, nullptr, HD);
        if (!p.noring) R[((p.pos0 + b) % HIST) * CONVW + p.off + i] = S[p.src + b * NE + i];
        float nv;
        if (p.push) { S[p.blk + b * p.blkstride + p.nb * NE + i] = cur[i] * rc; nv = oc; } else nv = cur[i] + oc;
        cur[i] = nv; S[p.ffn + b * NE + i] = nv;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    uint nb = p.nb + (p.push ? 1 : 0);
    /* mix over the blocks (one query) */
    float ss = 0; for (uint i = tid; i < NE; i += ntg) ss += cur[i] * cur[i];
    ss = tg_sum(ss, red, tid, ntg); float rc2 = rsqrt(ss / NE + 1e-6f);
    device const float *Q = M + p.frq;
    for (uint s = 0; s <= nb; s++) { float d = 0;
        if (s < nb) { device const float *st = S + p.blk + b * p.blkstride + s * NE; for (uint i = tid; i < NE; i += ntg) d += st[i] * Q[i]; }
        else for (uint i = tid; i < NE; i += ntg) d += cur[i] * rc2 * Q[i];
        d = tg_sum(d, red, tid, ntg); if (tid == 0) sc[s] = d * rsqrt((float)NE); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx = -INFINITY; for (uint s = 0; s <= nb; s++) mx = max(mx, sc[s]);
    float z = 0, w[24]; for (uint s = 0; s <= nb; s++) { w[s] = exp(sc[s] - mx); z += w[s]; }
    device float *x = S + p.xm + b * NE; float s2 = 0;
    for (uint i = tid; i < NE; i += ntg) { float a = 0; for (uint s = 0; s < nb; s++) a += w[s] * S[p.blk + b * p.blkstride + s * NE + i];
        a = (a + w[nb] * cur[i] * rc2) / z; x[i] = a; s2 += a * a; }
    if (!p.route) return;
    s2 = tg_sum(s2, red, tid, ntg); float r2 = rsqrt(s2 / NE + 1e-6f);
    for (uint i = tid; i < NE; i += ntg) x[i] = x[i] * r2 * M[p.nw + i];

}

// ---------------- DSpark drafter ----------------
// fused feature mix: row r < 3 -> layer-r context (dflare_norm[r]); row 3 -> markov context (dspark_norm)
struct PDFuse { uint feat, out, ostride_row, fuse, norm0, norm1, norm2, norm3, pos0, ctxr; };
kernel void k_dfuse(device float *S [[buffer(0)]], device const float *M [[buffer(1)]], constant PDFuse &p [[buffer(2)]],
                    uint gid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
    threadgroup float red[32];
    uint b = gid / 4, r = gid % 4;
    device const float *f = S + p.feat + b * 4 * NE; device const float *fw = M + p.fuse + r * 4;
    float mx = max(max(fw[0], fw[1]), max(fw[2], fw[3])), e0 = exp(fw[0] - mx), e1 = exp(fw[1] - mx), e2 = exp(fw[2] - mx), e3 = exp(fw[3] - mx), z = e0 + e1 + e2 + e3;
    device float *o = r == 3 ? S + p.ctxr + ((p.pos0 + b) % 64) * NE : S + p.out + r * p.ostride_row + b * NE; float ss = 0;
    for (uint i = tid; i < NE; i += ntg) { float v = (e0 * f[i] + e1 * f[NE + i] + e2 * f[2 * NE + i] + e3 * f[3 * NE + i]) / z; o[i] = v; ss += v * v; }
    ss = tg_sum(ss, red, tid, ntg); float rr = rsqrt(ss / NE + 1e-6f);
    uint nw = r == 0 ? p.norm0 : r == 1 ? p.norm1 : r == 2 ? p.norm2 : p.norm3;
    for (uint i = tid; i < NE; i += ntg) o[i] = o[i] * rr * M[nw + i];
}
// per-head rmsnorm (with weight) + NEOX rope at position pos0+b; optional fp16 store into a cache; one threadgroup of 128 per (b, head)
struct PRope { uint src, sstride, nheads, normw, has_norm, rope, invf, pos0, store, kind8, ntile, vsoff; };
kernel void k_headnorm_rope(device float *S [[buffer(0)]], device const float *M [[buffer(1)]], device half *C [[buffer(2)]], constant PRope &p [[buffer(3)]],
                            device char *Q8 [[buffer(4)]], device float *KS [[buffer(5)]], uint gid [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
    threadgroup float red[32]; threadgroup float v[HD];
    uint b = gid / p.nheads, h = gid % p.nheads;
    device float *x = S + p.src + b * p.sstride + h * HD;
    float a = x[d];
    if (p.has_norm) { float ss = tg_sum(a * a, red, d, HD); a = a * rsqrt(ss / HD + 1e-6f) * M[p.normw + d]; }
    v[d] = a; threadgroup_barrier(mem_flags::mem_threadgroup);
    if (p.rope) { uint i = d % (HD / 2); float th = (float)(p.pos0 + b) * M[p.invf + i]; float sn, cs; sn = precise::sincos(th, cs);
        float x0 = v[i], x1 = v[i + HD / 2]; a = d < HD / 2 ? x0 * cs - x1 * sn : x0 * sn + x1 * cs; }
    if (p.store) C[(size_t)(p.pos0 + b) * KVD + h * HD + d] = half(a); else x[d] = a;
    if (p.kind8) { uint pos = p.pos0 + b; float mx = tg_max(fabs(a), red, d, HD), sd = mx / 127.0f, iv = sd > 0 ? 1.0f / sd : 0.0f; char q = char(rint(a * iv));
        if (p.kind8 == 1) { Q8[((size_t)h * p.ntile * 32 + pos) * HD + d] = q; if (d == 0) KS[pos * NKV + h] = sd; }
        else { Q8[v8_off(h, p.ntile, pos, d)] = q; if (d == 0) KS[p.vsoff + pos * NKV + h] = sd; } }
}
struct PSwi { uint g, u, n; };
kernel void k_swiglu(device float *S [[buffer(0)]], constant PSwi &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return; float g = S[p.g + i]; S[p.g + i] = g / (1.0f + exp(-g)) * S[p.u + i];
}
struct PAdd { uint x, y, n; };
kernel void k_add(device float *S [[buffer(0)]], constant PAdd &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return; S[p.x + i] += S[p.y + i];
}

#define RANK 64
struct CMat { uint lo, hi; float gs; uint e2m1; char tbl[16]; };
struct PCtlF { uint cur, cbs, cjs, msg, mbs, mjs, B; CMat mt[9]; };
inline void ctl_lut(threadgroup half2 *lut, constant CMat &c, uint tid) {
    float lsc = c.e2m1 ? 1.0f : 0.0625f; lut[tid] = half2(half(c.tbl[tid & 15] * lsc), half(c.tbl[tid >> 4] * lsc));
}
// ---------------- controller in two dispatches ----------------
// A: one threadgroup per item (b, j): rms norms + r = ld [rms(c), rms(m)] -> S[hr + it*RANK]
struct PCtlA { uint cur, cbs, cjs, msg, mbs, mjs, J, hr; CMat mt[3]; };
kernel void k_ctl_a(device float *S [[buffer(0)]], device const uchar *W [[buffer(1)]], constant PCtlA &p [[buffer(2)]],
                    uint it [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half2 lut[256]; threadgroup float red[8][2]; threadgroup float rn[2];
    uint b = it / p.J, j = it % p.J; constant CMat &Mld = p.mt[j];
    ctl_lut(lut, Mld, tid);
    device const float *c = S + p.cur + b * p.cbs + j * p.cjs, *m = S + p.msg + b * p.mbs + j * p.mjs;
    float s1 = 0, s2 = 0; for (uint i = tid; i < NE; i += 256) { s1 += c[i] * c[i]; s2 += m[i] * m[i]; }
    s1 = simd_sum(s1); s2 = simd_sum(s2); if (lane == 0) { red[sg][0] = s1; red[sg][1] = s2; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 2) { float t = 0; for (uint g = 0; g < 8; g++) t += red[g][tid]; rn[tid] = rsqrt(t / NE + 1e-6f); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint KB = 2 * NE / 16; ulong woff = ((ulong)Mld.hi << 32) | Mld.lo; const float lsc = Mld.e2m1 ? 1.0f : 0.0625f;
    for (uint gi = 0; gi < 2; gi++) { uint g = sg + gi * 8; device const uchar *base = W + woff + (ulong)g * KB * 36;
        float acc[4] = {0, 0, 0, 0};
        for (uint kb = lane; kb < KB; kb += 32) {
            device const uchar *sb = base + (kb >> 2) * 144; device const uint4 *nw4 = (device const uint4 *)(sb + (kb & 3) * 32);
            uint4 NA = nw4[0], NB = nw4[1]; uint s4 = *(device const uint *)(sb + 128 + (kb & 3) * 4);
            uint k0 = kb * 16; bool isc = k0 < NE;
            device const float4 *xp = (device const float4 *)(isc ? c + k0 : m + (k0 - NE)); float r = rn[isc ? 0 : 1];
            float4 x0 = xp[0] * r, x1 = xp[1] * r, x2 = xp[2] * r, x3 = xp[3] * r;
            for (uint q = 0; q < 4; q++) { uint A = NA[q], Bw = NB[q];
                float sc = as_type<float>((((s4 >> (8 * q)) & 255u) << 20) + (117u << 23));
                half2 a0 = lut[A & 255], a1 = lut[(A >> 8) & 255], a2 = lut[(A >> 16) & 255], a3 = lut[A >> 24];
                half2 b0 = lut[Bw & 255], b1 = lut[(Bw >> 8) & 255], b2 = lut[(Bw >> 16) & 255], b3 = lut[Bw >> 24];
                acc[q] += sc * (dot(float4(a0.x, a1.x, a2.x, a3.x), x0) + dot(float4(a0.y, a1.y, a2.y, a3.y), x1)
                              + dot(float4(b0.x, b1.x, b2.x, b3.x), x2) + dot(float4(b0.y, b1.y, b2.y, b3.y), x3)); }
        }
        for (uint q = 0; q < 4; q++) { float v = simd_sum(acc[q]); if (lane == 0) S[p.hr + it * RANK + 4 * g + q] = v * (Mld.gs / lsc); }
    }
}
// B: grid J x 9; 256 threads = 64 row-groups x 4 K-blocks; g, f rows + gated combine for all B positions
struct PCtlB { uint cur, cbs, cjs, msg, mbs, mjs, J, hr, B; CMat mt[6]; };
kernel void k_ctl_b(device float *S [[buffer(0)]], device const uchar *W [[buffer(1)]], constant PCtlB &p [[buffer(2)]],
                    uint gid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half2 lut[2][256];
    uint j = gid / 9, sl = gid % 9; constant CMat &Mlg = p.mt[2 * j], &Mlf = p.mt[2 * j + 1];
    ctl_lut(lut[0], Mlg, tid); ctl_lut(lut[1], Mlf, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint g = sl * 64 + tid / 4, kb = tid % 4;
    float og[4][4], of[4][4]; for (uint a = 0; a < 4; a++) for (uint b = 0; b < 4; b++) { og[a][b] = 0; of[a][b] = 0; }
    for (uint mtx = 0; mtx < 2; mtx++) {
        constant CMat &Mx = mtx ? Mlf : Mlg; ulong woff = ((ulong)Mx.hi << 32) | Mx.lo; threadgroup const half2 *L = lut[mtx];
        device const uchar *sb = W + woff + (ulong)g * 4 * 36; device const uint4 *nw4 = (device const uint4 *)(sb + kb * 32);
        uint4 NA = nw4[0], NB = nw4[1]; uint s4 = *(device const uint *)(sb + 128 + kb * 4);
        for (uint q = 0; q < 4; q++) { uint A = NA[q], Bw = NB[q];
            float sc = as_type<float>((((s4 >> (8 * q)) & 255u) << 20) + (117u << 23));
            half2 a0 = L[A & 255], a1 = L[(A >> 8) & 255], a2 = L[(A >> 16) & 255], a3 = L[A >> 24];
            half2 b0 = L[Bw & 255], b1 = L[(Bw >> 8) & 255], b2 = L[(Bw >> 16) & 255], b3 = L[Bw >> 24];
            float4 w0 = float4(a0.x, a1.x, a2.x, a3.x), w1 = float4(a0.y, a1.y, a2.y, a3.y), w2 = float4(b0.x, b1.x, b2.x, b3.x), w3 = float4(b0.y, b1.y, b2.y, b3.y);
            for (uint b = 0; b < p.B; b++) { device const float4 *rv = (device const float4 *)(S + p.hr + (b * p.J + j) * RANK + kb * 16);
                float v = sc * (dot(w0, rv[0]) + dot(w1, rv[1]) + dot(w2, rv[2]) + dot(w3, rv[3]));
                if (mtx) of[q][b] += v; else og[q][b] += v; } }
    }
    const float lg_s = Mlg.gs / (Mlg.e2m1 ? 1.0f : 0.0625f), lf_s = Mlf.gs / (Mlf.e2m1 ? 1.0f : 0.0625f);
    for (uint q = 0; q < 4; q++) for (uint b = 0; b < 4; b++) {
        og[q][b] += simd_shuffle_xor(og[q][b], 1); og[q][b] += simd_shuffle_xor(og[q][b], 2);
        of[q][b] += simd_shuffle_xor(of[q][b], 1); of[q][b] += simd_shuffle_xor(of[q][b], 2); }
    if (kb == 0) for (uint b = 0; b < p.B; b++) { device float *c = S + p.cur + b * p.cbs + j * p.cjs; device const float *m = S + p.msg + b * p.mbs + j * p.mjs;
        for (uint q = 0; q < 4; q++) { uint i = 4 * g + q; float gg = og[q][b] * lg_s, ff = of[q][b] * lf_s;
            c[i] = (1.0f - tanh(ff)) * c[i] + tanh(gg) * m[i]; } }
}

// ======================= prefill GEMM (simdgroup matrix) =======================
// Y[row(t)][n] = sum_k W[n][k] * X[xrow(t)][k] for tokens t < T. Tile: 64 rows x 32 tokens, K step 64; 128 threads.
// The 4-bit tile is decoded once into threadgroup memory (half, scales folded in) and reused by all 32 tokens.
struct GJob { uint woff_lo, woff_hi, N, K; float gscale; uint e2m1; char tbl[16]; uint x, xs, y, ys, T; };	/* x: half rows at S + x, stride xs halves */
/* Y[t][n0..n0+63] = W[n0..][:] . X[xrow(t)] for TT tokens from t0. NT threads: 2 row halves x NT/64 token groups of
   simdgroups, each 32 rows x TT/(NT/64) tokens (TT == 8: simdgroups split the 64 rows). Per K step of 64 the 4-bit tile
   is decoded once into threadgroup memory (scales folded in) and the next step's raw superblocks are prefetched. */
template <uint TT, uint NT> void gemm_tile(device const uchar *W, device float *S, thread const GJob &j, uint n0, uint t0, uint T,
                      device const uint *xrow, uint tid, uint sg,
                      threadgroup half *sa, threadgroup half *sb, threadgroup float *so, threadgroup half *cb, threadgroup uint4 *raw) {
    constexpr uint NSG = NT / 32;
    constexpr bool T8 = TT == 8;
    constexpr uint TPS = T8 ? 8 : TT / (NSG / 2), QB = TPS / 8, RB = T8 ? 8 / NSG : 4, NPRE = (144 + NT - 1) / NT;
    const uint K = j.K, KB4 = K / 64; ulong woff = ((ulong)j.woff_hi << 32) | j.woff_lo;
    const float gs = j.gscale;
    simdgroup_float8x8 mc[RB][QB];
    for (uint i = 0; i < RB; i++) for (uint q = 0; q < QB; q++) mc[i][q] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint rsub = T8 ? sg * RB * 8 : (sg & 1) * 32, tsub = T8 ? 0 : (sg >> 1) * TPS;
    if (tid < 16) cb[tid] = half(j.tbl[tid]);
    uint4 pre[NPRE];
    #define LDRAW(kt_, i_) ((i_) < 144 && n0 + ((i_) / 9) * 4 < j.N ? ((device const uint4 *)(W + woff + ((ulong)(n0 / 4 + (i_) / 9) * KB4 + (kt_)) * 144))[(i_) % 9] : uint4(0))
    for (uint p = 0; p < NPRE; p++) pre[p] = LDRAW(0, tid + p * NT);
    for (uint kt = 0; kt < KB4; kt++) {
        for (uint p = 0; p < NPRE; p++) if (tid + p * NT < 144) raw[tid + p * NT] = pre[p];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kt + 1 < KB4) for (uint p = 0; p < NPRE; p++) pre[p] = LDRAW(kt + 1, tid + p * NT);
        /* weights: 256 units (row group, block, row) of 16 */
        for (uint u = tid; u < 256; u += NT) { uint g = u >> 4, kb = (u >> 2) & 3, rr = u & 3;
          threadgroup half4 *dst = (threadgroup half4 *)(sa + (g * 4 + rr) * 64 + kb * 16);
          if (n0 + g * 4 >= j.N) { dst[0] = dst[1] = dst[2] = dst[3] = half4(0); }
          else {
            threadgroup const uchar *sbk = (threadgroup const uchar *)(raw + g * 9);
            uint A = ((threadgroup const uint *)(sbk + kb * 32))[rr], Bw = ((threadgroup const uint *)(sbk + kb * 32 + 16))[rr];
            half hs = half(as_type<float>((((uint)sbk[128 + kb * 4 + rr]) << 20) + (117u << 23)) * gs);
            uchar4 a = as_type<uchar4>(A), b = as_type<uchar4>(Bw);
            dst[0] = half4(cb[a.x & 15], cb[a.y & 15], cb[a.z & 15], cb[a.w & 15]) * hs;
            dst[1] = half4(cb[a.x >> 4], cb[a.y >> 4], cb[a.z >> 4], cb[a.w >> 4]) * hs;
            dst[2] = half4(cb[b.x & 15], cb[b.y & 15], cb[b.z & 15], cb[b.w & 15]) * hs;
            dst[3] = half4(cb[b.x >> 4], cb[b.y >> 4], cb[b.z >> 4], cb[b.w >> 4]) * hs; } }
        /* activations: TT tokens x 64 k */
        for (uint e = tid; e < TT * 16; e += NT) { uint t = e >> 4, k4 = e & 15; half4 v = half4(0);
            if (t0 + t < T) { uint xr = xrow ? xrow[t0 + t] : t0 + t; v = *(device const half4 *)((device const half *)(S + j.x) + xr * j.xs + kt * 64 + k4 * 4); }
            ((threadgroup half4 *)sb)[t * 16 + k4] = v; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 64 && t0 + tsub < T; kk += 8) {	/* simdgroups past the last token skip the math */
            simdgroup_half8x8 ma[RB], mb[QB];
            for (uint i = 0; i < RB; i++) simdgroup_load(ma[i], sa + (rsub + i * 8) * 64 + kk, 64);
            for (uint q = 0; q < QB; q++) simdgroup_load(mb[q], sb + (tsub + q * 8) * 64 + kk, 64, ulong2(0, 0), true);
            for (uint i = 0; i < RB; i++) for (uint q = 0; q < QB; q++) simdgroup_multiply_accumulate(mc[i][q], ma[i], mb[q], mc[i][q]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (n0 + 64 <= j.N && t0 + TT <= T) {		/* full tile: straight to device memory */
        for (uint i = 0; i < RB; i++) for (uint q = 0; q < QB; q++)
            simdgroup_store(mc[i][q], S + j.y + (t0 + tsub + q * 8) * j.ys + n0 + rsub + i * 8, j.ys, ulong2(0, 0), true);
        return;
    }
    constexpr uint SH = TT < 64 ? TT : 64;		/* edge tile: stage SH tokens at a time */
    for (uint h = 0; h < TT; h += SH) {
        for (uint i = 0; i < RB; i++) for (uint q = 0; q < QB; q++) { uint tt = tsub + q * 8;
            if (tt >= h && tt < h + SH) simdgroup_store(mc[i][q], so + (tt - h) * 64 + rsub + i * 8, 64, ulong2(0, 0), true); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < SH * 64; e += NT) { uint t = e >> 6, n = e & 63;
            if (t0 + h + t < T && n0 + n < j.N) S[j.y + (t0 + h + t) * j.ys + n0 + n] = so[t * 64 + n]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
template <uint TT, uint NT> kernel void k_gemm(device const uchar *W [[buffer(0)]], device float *S [[buffer(1)]], constant GJob &j [[buffer(2)]],
                   uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float buf[(64 * 64 + TT * 64) / 2 > 64 * 64 ? (64 * 64 + TT * 64) / 2 : 64 * 64];	/* sa + sb, reused as the output stage */
    threadgroup half *sa = (threadgroup half *)buf, *sb = sa + 64 * 64; threadgroup half cbk[16]; threadgroup uint4 raw[144];
    GJob jl = j;
    gemm_tile<TT, NT>(W, S, jl, tg.x * 64, tg.y * TT, j.T, nullptr, tid, sg, sa, sb, buf, cbk, raw);
}
template [[host_name("k_gemm64")]] kernel void k_gemm<64, 128>(device const uchar *, device float *, constant GJob &, uint2, uint, uint);
template [[host_name("k_gemm128")]] kernel void k_gemm<128, 256>(device const uchar *, device float *, constant GJob &, uint2, uint, uint);

#ifndef MT
#define MT 32         /* tokens per expert GEMM tile */
#endif
// ---------------- prefill MoE: counting-sort planner (one threadgroup of 1024) ----------------
struct PPlanB { uint B, ids, slotmap, slottok, tiles, mt, big; };	/* big-token tiles first (list at tiles), remainders in mt-token tiles (list at tiles + 3072) */
kernel void k_moe_plan_big(device float *S [[buffer(0)]], device uint *ARGS [[buffer(1)]], constant PPlanB &p [[buffer(2)]],
                           uint tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint cnt[NEXP], fill[NEXP]; threadgroup uint off[NEXP];
    device const uint *ids = (device const uint *)(S + p.ids);
    device uint *smap = (device uint *)(S + p.slotmap), *stok = (device uint *)(S + p.slottok), *tl = (device uint *)(S + p.tiles);
    const uint ns = p.B * NUSED;
    if (tid < NEXP) { atomic_store_explicit(&cnt[tid], 0, memory_order_relaxed); atomic_store_explicit(&fill[tid], 0, memory_order_relaxed); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < ns; i += 1024) { uint e = ids[i]; if (e < NEXP) atomic_fetch_add_explicit(&cnt[e], 1, memory_order_relaxed); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { uint o = 0, nb = 0, nt = 0;
        for (uint e = 0; e < NEXP; e++) { uint c = atomic_load_explicit(&cnt[e], memory_order_relaxed), s0 = 0; off[e] = o;
            for (; p.big && s0 < c; s0 += p.big) { tl[nb * 3] = e; tl[nb * 3 + 1] = o + s0; tl[nb * 3 + 2] = min(p.big, c - s0); nb++; }
            for (; s0 < c; s0 += p.mt) { tl[3072 + nt * 3] = e; tl[3072 + nt * 3 + 1] = o + s0; tl[3072 + nt * 3 + 2] = min(p.mt, c - s0); nt++; }
            o += c; }
        for (uint ph = 0; ph < 2; ph++) { uint rows = ph ? (NE + 63) / 64 : (2 * NFF + 63) / 64;	/* per phase: big list, small list */
            ARGS[ph * 6] = rows; ARGS[ph * 6 + 1] = nb; ARGS[ph * 6 + 2] = 1; ARGS[ph * 6 + 3] = rows; ARGS[ph * 6 + 4] = nt; ARGS[ph * 6 + 5] = 1; } }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < ns; i += 1024) { uint e = ids[i];
        if (e < NEXP) { uint pos = off[e] + atomic_fetch_add_explicit(&fill[e], 1, memory_order_relaxed); stok[pos] = i / NUSED; smap[i] = pos; } }
}
// grouped expert GEMM: grid (row tiles, token tiles); phase 0 = gate_up (gather token rows), 1 = down (SITU on gate_up rows)
struct PMoeG { uint layer, phase, xm, gu, dn, slottok, tiles, act; };
template <uint TMT, uint NT> kernel void k_gemm_moe(device const uchar *W [[buffer(0)]], device float *S [[buffer(1)]], device const ETab *ET [[buffer(2)]],
                       constant PMoeG &p [[buffer(3)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float buf[(64 * 64 + TMT * 64) / 2 > 64 * 64 ? (64 * 64 + TMT * 64) / 2 : 64 * 64];	/* sa + sb, reused as the output stage */
    threadgroup half *sa = (threadgroup half *)buf, *sb = sa + 64 * 64; threadgroup half cbk[16]; threadgroup uint4 raw[144];
    device const uint *tl = (device const uint *)(S + p.tiles) + tg.y * 3;
    uint e = tl[0], start = tl[1], count = tl[2];
    device const ETab &et = ET[p.layer * NEXP + e];
    GJob j; for (uint i = 0; i < 16; i++) j.tbl[i] = et.tbl[i];
    j.e2m1 = et.e2m1; j.T = count;
    device const uint *xr = nullptr;
    if (p.phase == 0) { j.woff_lo = et.gu_lo; j.woff_hi = et.gu_hi; j.N = 2 * NFF; j.K = NE; j.gscale = et.gsg;
        j.x = p.xm; j.xs = NE; xr = (device const uint *)(S + p.slottok) + start; j.y = p.gu + start * 2 * NFF; j.ys = 2 * NFF; }	/* xm, act: half */
    else { j.woff_lo = et.dn_lo; j.woff_hi = et.dn_hi; j.N = NE; j.K = NFF; j.gscale = et.gsd;
        j.x = p.act + start * NFF / 2; j.xs = NFF; j.y = p.dn + start * NE; j.ys = NE; }
    if (tg.x * 64 >= j.N) return;
    gemm_tile<TMT, NT>(W, S, j, tg.x * 64, 0, count, xr, tid, sg, sa, sb, buf, cbk, raw);
}
template [[host_name("k_gemm_moe8")]] kernel void k_gemm_moe<8, 128>(device const uchar *, device float *, device const ETab *, constant PMoeG &, uint2, uint, uint);
template [[host_name("k_gemm_moe32")]] kernel void k_gemm_moe<32, 128>(device const uchar *, device float *, device const ETab *, constant PMoeG &, uint2, uint, uint);
template [[host_name("k_gemm_moe128")]] kernel void k_gemm_moe<128, 256>(device const uchar *, device float *, device const ETab *, constant PMoeG &, uint2, uint, uint);

// ---------------- prefill router: fp32 GEMM with simdgroup float matrices ----------------
// L[t][e] = sum_k Wr[e][k] X[t][k]; tile 64 experts x 32 tokens, K step 32; 128 threads
struct PRouterG { uint w, x, y, T; };
kernel void k_router_gemm(device float *S [[buffer(0)]], device const float *M [[buffer(1)]], constant PRouterG &p [[buffer(2)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sw[64 * 32], sx[32 * 32], so[32 * 64];
    const uint e0 = tg.x * 64, t0 = tg.y * 32, rsub = (sg & 1) * 32, tsub = (sg >> 1) * 16;
    simdgroup_float8x8 mc[4][2];
    for (uint i = 0; i < 4; i++) for (uint q = 0; q < 2; q++) mc[i][q] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = 0; k0 < NE; k0 += 32) {
        for (uint e = tid; e < 64 * 32; e += 128) { uint r = e >> 5, k = e & 31; sw[e] = M[p.w + (e0 + r) * NE + k0 + k]; }
        for (uint e = tid; e < 32 * 32; e += 128) { uint t = e >> 5, k = e & 31; sx[e] = t0 + t < p.T ? S[p.x + (t0 + t) * NE + k0 + k] : 0.0f; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_float8x8 ma[4], mb[2];
            for (uint i = 0; i < 4; i++) simdgroup_load(ma[i], sw + (rsub + i * 8) * 32 + kk, 32);
            for (uint q = 0; q < 2; q++) simdgroup_load(mb[q], sx + (tsub + q * 8) * 32 + kk, 32, ulong2(0, 0), true);
            for (uint i = 0; i < 4; i++) for (uint q = 0; q < 2; q++) simdgroup_multiply_accumulate(mc[i][q], ma[i], mb[q], mc[i][q]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 4; i++) for (uint q = 0; q < 2; q++) simdgroup_store(mc[i][q], so + (tsub + q * 8) * 64 + rsub + i * 8, 64, ulong2(0, 0), true);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < 32 * 64; e += 128) { uint t = e >> 6, n = e & 63; if (t0 + t < p.T) S[p.y + (t0 + t) * NEXP + e0 + n] = so[e]; }
}

// ---------------- prefill flash attention (fp16 KV cache, simdgroup matrices) ----------------
// one threadgroup (128) per (kv head, block of 16 query tokens = 48 query rows); key blocks of 32; online softmax;
// relative bias, causal / sliding window, sink and sigmoid gate applied here (no partials / merge).
struct PAttnP { uint T, pos0, cap, ring, swa, q, r, relb, sink, gate, att, has_bias; };
kernel void k_attn_prefill(device const half *K [[buffer(0)]], device const half *V [[buffer(1)]], device float *S [[buffer(2)]],
                           device const float *M [[buffer(3)]], constant PAttnP &p [[buffer(4)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint kh = tg.x, tb0 = tg.y * 16;               /* query tokens tb0 .. tb0+15; row r = tok*3 + head-in-group */
    const uint ntok = min(16u, p.T - tb0);
    threadgroup half qs[48 * HD];                         /* 12 KB */
    threadgroup half kv[32 * HD];                         /* 8 KB: K block, then V block */
    threadgroup float sc[48 * 32];                        /* 6 KB scores / probabilities */
    threadgroup half ph[48 * 32];                         /* 3 KB */
    threadgroup float mrow[48], lrow[48], arow[48];
    threadgroup float dg[8 * 64];                         /* diag(alpha) for 6 row blocks (8x8 each) */
    const float isq = rsqrt((float)HD);
    for (uint i = tid; i < 48 * HD; i += 256) { uint r = i / HD, d = i % HD, t = r / 3, h = kh * 3 + r % 3;
        qs[i] = t < ntok ? half(S[p.q + ((tb0 + t) * NH + h) * HD + d] * isq) : half(0); }
    if (tid < 48) { mrow[tid] = -INFINITY; lrow[tid] = 0; }
    for (uint i = tid; i < 8 * 64; i += 256) dg[i] = 0;
    /* O accumulators: 6 row blocks x 16 dim blocks = 96; simdgroup sg owns dim blocks sg*4 .. sg*4+3 for all 6 row blocks */
    simdgroup_float8x8 oc[6][2];      /* simdgroup sg owns dim blocks 2sg, 2sg+1 */
    for (uint i = 0; i < 6; i++) for (uint j = 0; j < 2; j++) oc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const int qfirst = (int)(p.pos0 + tb0), qlast = qfirst + (int)ntok - 1;
    const int kbeg = p.swa ? max(0, qfirst - 1023) : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k0 = kbeg & ~31; k0 <= qlast; k0 += 32) {
        /* K block */
        for (uint i = tid; i < 32 * HD / 4; i += 256) { uint t = i / (HD / 4), d4 = i % (HD / 4); int pp = k0 + (int)t;
            half4 v = half4(0); if (pp >= kbeg && pp <= qlast) { uint sp = p.ring ? (uint)pp % p.cap : (uint)pp; v = ((device const half4 *)(K + (size_t)sp * KVD + kh * HD))[d4]; }
            ((threadgroup half4 *)kv)[i] = v; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        /* S = Q K^T: 6 x 4 blocks; simdgroup sg -> key block sg (8 keys), all 6 row blocks */
        { const uint kbk = sg & 3, rh = (sg >> 2) * 3;     /* key block, row blocks rh..rh+2 */
          simdgroup_float8x8 s3[3]; for (uint i = 0; i < 3; i++) s3[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
          for (uint kk = 0; kk < HD; kk += 8) { simdgroup_half8x8 kb; simdgroup_load(kb, kv + (kbk * 8) * HD + kk, HD, ulong2(0, 0), true);
              for (uint i = 0; i < 3; i++) { simdgroup_half8x8 qa; simdgroup_load(qa, qs + ((rh + i) * 8) * HD + kk, HD); simdgroup_multiply_accumulate(s3[i], qa, kb, s3[i]); } }
          for (uint i = 0; i < 3; i++) simdgroup_store(s3[i], sc + ((rh + i) * 8) * 32 + kbk * 8, 32); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        /* bias + mask (all threads), then online softmax per row (simdgroup per row, lane per key) */
        for (uint e = tid; e < 48 * 32; e += 256) { uint r = e >> 5, c = e & 31, t = r / 3, h = kh * 3 + r % 3;
            int qp = qfirst + (int)t, pp = k0 + (int)c, dist = qp - pp; float v = sc[e];
            bool ok = t < ntok && pp >= kbeg && dist >= 0 && !(p.swa && dist >= 1024);
            if (ok && p.has_bias && dist < REL_EXT) { device const float4 *rb = (device const float4 *)(M + p.relb + dist * NREL);
                device const float4 *rq = (device const float4 *)(S + p.r + ((tb0 + t) * NH + h) * NREL);
                v += dot(rq[0], rb[0]) + dot(rq[1], rb[1]) + dot(rq[2], rb[2]) + dot(rq[3], rb[3]); }
            sc[e] = ok ? v : -INFINITY; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = sg; r < 48; r += 8) { float v = sc[r * 32 + lane], mx = simd_max(v);
            float mo = mrow[r], mn = max(mo, mx), a = mn == -INFINITY ? 1.0f : exp(mo - mn);
            float e = mn == -INFINITY ? 0.0f : exp(v - mn); ph[r * 32 + lane] = half(e); float l = simd_sum(e);
            if (lane == 0) { mrow[r] = mn; lrow[r] = lrow[r] * a + l; arow[r] = a; } }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        /* V block (reuse kv) + diag(alpha) */
        for (uint i = tid; i < 32 * HD / 4; i += 256) { uint t = i / (HD / 4), d4 = i % (HD / 4); int pp = k0 + (int)t;
            half4 v = half4(0); if (pp >= kbeg && pp <= qlast) { uint sp = p.ring ? (uint)pp % p.cap : (uint)pp; v = ((device const half4 *)(V + (size_t)sp * KVD + kh * HD))[d4]; }
            ((threadgroup half4 *)kv)[i] = v; }
        for (uint i = tid; i < 48; i += 256) { uint blk = i / 8, rr = i % 8; dg[blk * 64 + rr * 8 + rr] = arow[i]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        /* O = diag(alpha) O + P V */
        for (uint i = 0; i < 6; i++) { simdgroup_float8x8 da; simdgroup_load(da, dg + i * 64, 8);
            for (uint j = 0; j < 2; j++) { simdgroup_float8x8 t0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                simdgroup_multiply_accumulate(t0, da, oc[i][j], t0); oc[i][j] = t0; } }
        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_half8x8 pa[6]; for (uint i = 0; i < 6; i++) simdgroup_load(pa[i], ph + (i * 8) * 32 + kk, 32);
            for (uint j = 0; j < 2; j++) { simdgroup_half8x8 vb; simdgroup_load(vb, kv + kk * HD + (sg * 2 + j) * 8, HD);
                for (uint i = 0; i < 6; i++) simdgroup_multiply_accumulate(oc[i][j], pa[i], vb, oc[i][j]); } }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    /* epilogue: stage O through kv/sc memory in two halves of rows (48 x 128 floats = 24 KB > staging): write per row block */
    for (uint i = 0; i < 6; i++) {
        for (uint j = 0; j < 2; j++) simdgroup_store(oc[i][j], sc + (sg * 2 + j) * 8, HD);   /* 8 rows x 128 into sc (as [8][128]) */
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < 8 * HD; e += 256) { uint rr = e / HD, d = e % HD, r = i * 8 + rr, t = r / 3, h = kh * 3 + r % 3;
            if (t < ntok) { uint b = tb0 + t; float z = lrow[r] + exp(S[p.sink + b * NH + h] - mrow[r]);
                float o = sc[rr * HD + d] / z; o /= 1.0f + exp(-S[p.gate + b * QD + h * HD + d]);
                S[p.att + b * QD + h * HD + d] = o; } }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

struct PSituA { uint gu, act, n; };	/* act: half */
kernel void k_situ_all(device float *S [[buffer(0)]], constant PSituA &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return; uint r = i / NFF, c = i % NFF;
    float g = S[p.gu + r * 2 * NFF + c], up = S[p.gu + r * 2 * NFF + NFF + c];
    ((device half *)(S + p.act))[r * NFF + c] = half(SITU_BETA * tanh(g / SITU_BETA) / (1.0f + exp(-g)) * (SITU_LB * tanh(up / SITU_LB)));
}
// rows of fp32 -> fp16 (GEMM inputs): dst half rows of n, stride n
struct PHalf { uint src, ss, dst, n, rows; };
kernel void k_tohalf(device float *S [[buffer(0)]], constant PHalf &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n / 4 * p.rows) return; uint r = i / (p.n / 4), c = i % (p.n / 4);
    ((device half4 *)(S + p.dst))[i] = half4(*(device const float4 *)(S + p.src + r * p.ss + c * 4));
}
