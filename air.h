/* air.h - Inkling 2 Mini: model constants, weight file layout, gpu interface. */
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

enum {
	NE = 2304, NH = 24, NKV = 8, HD = 128, QD = NH * HD, KVD = NKV * HD, NREL = 16, REL = 1024,
	NEXP = 128, NUSED = 8, NFF = 1152, RANK = 64, BLOCK = 6,
	NPHYS = 24, NVIS = 48, CORE0 = 6, CORE1 = 18, NPASS = 3,
	SWA = 1024, HIST = 16, CONVW = 2 * KVD + 2 * NE,
	NVOCAB = 200058, NVOCAB4 = 200060,
	MAXB = 16, PMAX = 2048, TOPC = 16,
	/* drafter */
	DL = 3, DBLOCK = 8, DFF = 9216, DRANK = 256, NFEAT = 4, CTXRING = 64, DCAND = 64,
};
#define RING (SWA + PMAX + 32)	/* sliding window ring: prefill writes PMAX rows before attending */
#define EPS 1e-6f
#define QK_CLIP 22.627417f
#define REL_CLIP 8.0f
#define WSCALE 8.0f

/* special tokens */
enum { T_USER = 200000, T_MODEL, T_SYS, T_TOOL, T_TEXT, T_IMAGE, T_EOS, T_THINK = 200008, T_END = 200010,
       T_XML = 200024, T_TOOLJSON = 200049, T_TOOLTEXT = 200057 };

/*
 * 4-bit matrix (NV4R). Rows in groups of 4, K in blocks of 16, 4 blocks per 144-byte superblock:
 * 4 x 32 bytes of nibbles (byte b of [0,16): lo -> row b/4, k 4j+b%4 for j=0, hi j=1; [16,32): j=2,3)
 * then 16 scale bytes (E5M3, block kb row r at 4*kb+r). weight = cb[nibble] * scale * gs.
 * cb is E2M1 x 2 for the NVFP4 experts, IQ4_NL for everything quantized from bf16.
 */
typedef struct { uint64_t off; uint32_t N, K, nl; float gs; } mat;

/* fp32 "smalls" are addressed by float offset */
typedef struct {
	mat q, k, v, gate, o, rel, sink, kdyn, vdyn, ld[4], lg[4], lf[4], gu[NEXP], dn[NEXP];
	uint32_t anorm, qn, kn, vn, fnorm, qclip, rclip, relb, kconv, vconv, apc, fpc, arq, narq, frq, ebias, router;
} layer;
typedef struct {
	mat q, k, v, o, kc, vc, up, gate, down;
	uint32_t an, qn, kn, kcn, fn, dfl, swa;
} dlayer;
typedef struct {
	char magic[8];
	uint64_t smalls, nsmalls, arena, narena, emb, unemb, w1, joint, hout, vocab, nvocab;
	uint32_t enorm, onorm, oq, loopq, loopkvq, fuse, dnorm, doutn, mask, jointb, houtb;
	uint8_t swa[NVIS];
	mat out;
	layer L[NPHYS];
	dlayer D[DL];
} wfile;
#define AIRMAGIC "air1\0\0\0"
/* q8 rows (embeddings, exact lm head): NE int8 + NE/32 fp16 scales */
#define Q8ROW (NE + NE / 16)

/* weights in memory (air.c) */
extern wfile *W;
extern float *SM;
extern uint8_t *AR;
void q8row(uint64_t off, int row, float *out);

/* gpu.m */
int gpu_init(int max_ctx);
void gpu_forward(const int *tok, int B, int pos0, int last, int *cand);	/* cand [B][TOPC] (last: only row B-1) */
void gpu_prefill(const int *tok, int T, int pos0, int *cand);	/* cand [TOPC], last row only */
const float *gpu_x(void);					/* final hidden rows of the last call */
void gpu_draft(int ipos, int ni, int anchor, int pos0, int nd, int *cand, const float **x);
float *gpu_ctxring(void);
float *gpu_conv(int vis);
int gpu_state(FILE *f, int n, int wr);			/* save / load the first n positions */
