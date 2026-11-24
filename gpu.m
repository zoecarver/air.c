/* gpu.m - Metal host: buffers, the 48-visit forward (decode and prefill), the DSpark drafter. */
#import <Metal/Metal.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "air.h"

static const char msrc[] = {
#embed "air.metal"
, 0 };

static const signed char E2[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };
static const signed char NL[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

static id<MTLDevice> dev;
static id<MTLCommandQueue> queue;
static id<MTLComputeCommandEncoder> E;
static id<MTLBuffer> bW, bM, bS, bET, bJobs, bCnt, bArgs, bCand, bDCand;
static id<MTLBuffer> bK[NVIS], bV[NVIS], bR[NVIS], bK8[NVIS], bV8[NVIS], bKS[NVIS];
static id<MTLBuffer> bDK[DL], bDV[DL], bDK8[DL], bDV8[DL], bDKS[DL];
static float *S;
static int cap[NVIS], ntile[NVIS], dcap, dntile, njobs, ncnt;
static uint32_t invf, mfree;

#define PIPES X(copy) X(rmsnorm) X(mix) X(ctl_prep) X(ctl_combine) X(ctl_a) X(ctl_b) X(gemv) X(ring_write) X(conv_kv) \
	X(qr) X(attn) X(attn8) X(attn_merge) X(post) X(ffn_front) X(topk) X(moe_plan) X(moe_combine) X(top_a) X(top_b) \
	X(dfuse) X(headnorm_rope) X(swiglu) X(add) X(gemm) X(moe_plan_big) X(router_gemm) X(attn_prefill) X(situ_all)
#define X(n) static id<MTLComputePipelineState> P_##n;
PIPES
#undef X
static id<MTLComputePipelineState> P_mm[5], P_mmbig[5], P_mmmoe, P_gmoe8, P_gmoe32;

/* scratch layout (float offsets), one for decode (MAXB rows) and one for prefill (PMAX rows) */
typedef struct {
	uint32_t cur, ffn, prec, bankl, blk, preb, bank, msg, inp, hr, h2, cg, cf, q, kk, vv, r, gate, sink, dk, dv,
		att, o, xm, ym, log, ids, wts, gu, act, dn, smap, pm, pl, pacc, x, lg, part, feat, stok, tiles;
} lay;
static lay LD, LP, *Y = &LD;
static uint32_t DX, DH, DQ, DKK, DVV, DATT, DO, DG, DU, DC, CTXR;
enum { BLKS = 16 * NE, ATTCH = 512, NTOPCH = 64, MAXJ = 160, JOBCAP = 60000, CNTCAP = 8192 };

typedef struct { uint32_t lo, hi, N, K; float gs; uint32_t B, x0, x1, x2, x3, y, ys, tile0; signed char tbl[16]; uint32_t e2m1, act, ngi; } job;
typedef struct { uint32_t gulo, guhi, dnlo, dnhi; float gsg, gsd; uint32_t e2m1; signed char tbl[16]; uint32_t pad; } etab;
typedef struct { uint32_t lo, hi, N, K; float gs; uint32_t e2m1; signed char tbl[16]; uint32_t x, xs, y, ys, T, xidx, yidx, act; } gjob;
typedef struct { uint32_t lo, hi; float gs; uint32_t e2m1; signed char tbl[16]; } cmat;

static size_t sused;
static uint32_t salloc(size_t n) { uint32_t o = (uint32_t)sused; sused += (n + 63) & ~(size_t)63; return o; }
static void layout(lay *l, int mb, int lgrows, int slots) {
	l->cur = salloc(mb * NE); l->ffn = salloc(mb * NE); l->prec = salloc(mb * NE); l->bankl = salloc(mb * NE);
	l->blk = salloc((size_t)mb * BLKS); l->preb = salloc((size_t)mb * BLKS); l->bank = salloc((size_t)mb * BLKS);
	l->msg = salloc(mb * 4 * NE); l->inp = salloc(mb * 3 * NE); l->h2 = salloc(mb * 6 * NE); l->hr = salloc(mb * 3 * RANK);
	l->cg = salloc(mb * 3 * NE); l->cf = salloc(mb * 3 * NE); l->q = salloc(mb * QD); l->kk = salloc(mb * KVD); l->vv = salloc(mb * KVD);
	l->r = salloc(mb * NH * NREL); l->gate = salloc(mb * QD); l->sink = salloc(mb * NH); l->dk = salloc(mb * 32); l->dv = salloc(mb * 32);
	l->att = salloc(mb * QD); l->o = salloc(mb * NE); l->xm = salloc(mb * NE); l->ym = salloc(mb * NE); l->log = salloc(mb * NEXP);
	l->ids = salloc(mb * NUSED); l->wts = salloc(mb * NUSED); l->gu = salloc((size_t)slots * 2 * NFF); l->act = salloc((size_t)slots * NFF);
	l->dn = salloc((size_t)slots * NE); l->smap = salloc(mb * NUSED); l->pm = salloc(NKV * ATTCH * 24); l->pl = salloc(NKV * ATTCH * 24);
	l->pacc = salloc((size_t)NKV * ATTCH * 24 * HD); l->x = salloc(mb * NE); l->lg = salloc((size_t)lgrows * (NVOCAB + 3));
	l->part = salloc(MAXB * NTOPCH * 64 * 2); l->feat = salloc(mb * 4 * NE); l->stok = salloc(mb * NUSED); l->tiles = salloc(3 * 2048);
}

static id<MTLComputePipelineState> mkpipe(id<MTLLibrary> lib, const char *name) {
	NSError *e = nil;
	id<MTLFunction> f = [lib newFunctionWithName:@(name)];
	id<MTLComputePipelineState> p = f ? [dev newComputePipelineStateWithFunction:f error:&e] : nil;
	if (!p) { fprintf(stderr, "air: kernel %s: %s\n", name, e ? e.description.UTF8String : "missing"); exit(1); }
	return p;
}
static id<MTLBuffer> buf(size_t n) { return [dev newBufferWithLength:n ? n : 16 options:MTLResourceStorageModeShared]; }

int gpu_init(int max_ctx) {
	@autoreleasepool {
	dev = MTLCreateSystemDefaultDevice(); queue = [dev newCommandQueue];
	NSError *e = nil;
	id<MTLLibrary> lib = [dev newLibraryWithSource:@(msrc) options:nil error:&e];
	if (!lib) { fprintf(stderr, "air: metal: %s\n", e.description.UTF8String); return -1; }
#define X(n) P_##n = mkpipe(lib, "k_" #n);
	PIPES
#undef X
	for (int b = 1; b <= 4; b++) { char n[16]; snprintf(n, sizeof n, "mm4_%d", b); P_mm[b] = mkpipe(lib, n); snprintf(n, sizeof n, "mm4big_%d", b); P_mmbig[b] = mkpipe(lib, n); }
	P_mmmoe = mkpipe(lib, "mm4_moe"); P_gmoe8 = mkpipe(lib, "k_gemm_moe8"); P_gmoe32 = mkpipe(lib, "k_gemm_moe32");

	bW = [dev newBufferWithBytesNoCopy:AR length:(W->narena + 16383) & ~(size_t)16383 options:MTLResourceStorageModeShared deallocator:nil];
	if (!bW) { fprintf(stderr, "air: cannot map weights\n"); return -1; }
	bM = buf((W->nsmalls + HD) * 4); memcpy(bM.contents, SM, W->nsmalls * 4);
	invf = (uint32_t)W->nsmalls; mfree = invf + HD / 2;
	for (int i = 0; i < HD / 2; i++) ((float *)bM.contents)[invf + i] = (float)pow(1e6, -2.0 * i / HD);
	SM = bM.contents;

	sused = 0;
	layout(&LD, MAXB, MAXB, 128);
	layout(&LP, PMAX, 1, PMAX * NUSED);
	DX = salloc(DBLOCK * NE); DH = salloc(DBLOCK * NE); DQ = salloc(DBLOCK * QD); DKK = salloc(DBLOCK * KVD); DVV = salloc(DBLOCK * KVD);
	DATT = salloc(DBLOCK * QD); DO = salloc(DBLOCK * NE); DG = salloc(DBLOCK * DFF); DU = salloc(DBLOCK * DFF);
	DC = salloc(3 * MAXB * NE); CTXR = salloc(CTXRING * NE);
	bS = buf(sused * 4); S = bS.contents;

	bET = buf(sizeof(etab) * NPHYS * NEXP);
	for (int i = 0; i < NPHYS; i++) for (int x = 0; x < NEXP; x++) {
		const mat *g = &W->L[i].gu[x], *d = &W->L[i].dn[x]; etab *t = (etab *)bET.contents + i * NEXP + x;
		*t = (etab){ (uint32_t)g->off, (uint32_t)(g->off >> 32), (uint32_t)d->off, (uint32_t)(d->off >> 32), g->gs, d->gs, !g->nl };
		memcpy(t->tbl, g->nl ? NL : E2, 16);
	}
	bJobs = buf(sizeof(job) * JOBCAP); bCnt = buf(4 * CNTCAP); bArgs = buf(4 * 3 * 2 * NVIS);
	bCand = buf(4 * MAXB * TOPC); bDCand = buf(4 * DBLOCK * DCAND);
	for (int v = 0; v < NVIS; v++) {
		cap[v] = W->swa[v] ? (max_ctx < RING ? max_ctx : RING) : max_ctx;
		bK[v] = buf((size_t)cap[v] * KVD * 2); bV[v] = buf((size_t)cap[v] * KVD * 2); bR[v] = buf(HIST * CONVW * 4);
		if (!W->swa[v]) { ntile[v] = cap[v] / 32 + 1;
			bK8[v] = buf((size_t)ntile[v] * 32 * KVD); bV8[v] = buf((size_t)ntile[v] * 32 * KVD); bKS[v] = buf((size_t)cap[v] * NKV * 8); }
	}
	dcap = max_ctx + DBLOCK; dntile = dcap / 32 + 1;
	for (int i = 0; i < DL; i++) {
		bDK[i] = buf((size_t)dcap * KVD * 2); bDV[i] = buf((size_t)dcap * KVD * 2);
		if (!W->D[i].swa) { bDK8[i] = buf((size_t)dntile * 32 * KVD); bDV8[i] = buf((size_t)dntile * 32 * KVD); bDKS[i] = buf((size_t)dcap * NKV * 8); }
	}
	/* keep everything resident: avoids per-commit residency work on the 14 GB weight buffer */
	MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new]; rd.initialCapacity = 512;
	id<MTLResidencySet> rs = [dev newResidencySetWithDescriptor:rd error:&e];
	if (rs) {
		id<MTLBuffer> all[] = { bW, bM, bS, bET, bJobs, bCnt, bArgs, bCand, bDCand };
		for (unsigned i = 0; i < sizeof all / sizeof *all; i++) [rs addAllocation:all[i]];
		for (int v = 0; v < NVIS; v++) for (id<MTLBuffer> b in @[bK[v], bV[v], bR[v]]) [rs addAllocation:b];
		for (int v = 0; v < NVIS; v++) if (bK8[v]) { [rs addAllocation:bK8[v]]; [rs addAllocation:bV8[v]]; [rs addAllocation:bKS[v]]; }
		for (int i = 0; i < DL; i++) { [rs addAllocation:bDK[i]]; [rs addAllocation:bDV[i]];
			if (bDK8[i]) { [rs addAllocation:bDK8[i]]; [rs addAllocation:bDV8[i]]; [rs addAllocation:bDKS[i]]; } }
		[rs commit]; [rs requestResidency]; [queue addResidencySet:rs];
	}
	}
	return 0;
}

/* ---------------- encoding helpers ---------------- */
static void set(id<MTLComputePipelineState> p) { [E setComputePipelineState:p]; }
static void arg(int i, id<MTLBuffer> b) { [E setBuffer:b offset:0 atIndex:i]; }
#define P(i, x) [E setBytes:&(x) length:sizeof(x) atIndex:i]
static void grid(int ntg, int nt) { [E dispatchThreadgroups:MTLSizeMake(ntg, 1, 1) threadsPerThreadgroup:MTLSizeMake(nt, 1, 1)]; }
static void grid2(int x, int y, int nt) { [E dispatchThreadgroups:MTLSizeMake(x, y, 1) threadsPerThreadgroup:MTLSizeMake(nt, 1, 1)]; }
static void threads(int n) { grid((n + 255) / 256, 256); }
static int phys(int v) { return v < CORE0 ? v : v < CORE0 + NPASS * (CORE1 - CORE0) ? CORE0 + (v - CORE0) % (CORE1 - CORE0) : v - (NPASS - 1) * (CORE1 - CORE0); }

static void copy(uint32_t src, uint32_t dst, int n, int nvec, int ss, int ds) {
	struct { uint32_t src, dst, n, nvec, ss, ds; } p = { src, dst, n, nvec, ss, ds };
	set(P_copy); arg(0, bS); P(1, p); threads(n * nvec);
}
static void norm(uint32_t in, uint32_t out, int w, int nvec, int is, int os) {
	struct { uint32_t in, out, w, hasw, n, is, os; float eps; } p = { in, out, w < 0 ? 0 : w, w >= 0, NE, is, os, EPS };
	set(P_rmsnorm); arg(0, bS); arg(1, bM); P(2, p); grid(nvec, 512);
}
/* depth mix over nb blocks + cur with nq queries; optional per-query rmsnorm weights */
static void mix(int nb, uint32_t blk, uint32_t cur, uint32_t q, int nq, uint32_t out, int os, int oqs, int B, int n0, int n1, int n2) {
	struct { uint32_t nb, blk, bs, cur, cs, q, nq, out, os, oqs, n0, n1, n2, hasn; } p =
		{ nb, blk, BLKS, cur, NE, q, nq, out, os, oqs, n0 < 0 ? 0 : n0, n1 < 0 ? 0 : n1, n2 < 0 ? 0 : n2, n0 >= 0 };
	set(P_mix); arg(0, bS); arg(1, bM); P(2, p); grid(B * nq, 512);
}
static void ring(int v, uint32_t src, int ss, int off, int nch, int pos0, int B) {
	struct { uint32_t src, ss, off, nch, pos0, B; } p = { src, ss, off, nch, pos0, B };
	set(P_ring_write); arg(0, bS); arg(1, bR[v]); P(2, p); threads(B * nch);
}
static void gemv(uint32_t w, int N, int K, uint32_t x, int xs, uint32_t y, int ys, int B) {
	struct { uint32_t w, N, K, x, xs, y, ys, B; } p = { w, N, K, x, xs, y, ys, B };
	set(P_gemv); arg(0, bS); arg(1, bM); P(2, p); grid((N + 7) / 8, 256);
}

/* decode matmuls: a list of jobs (matrix, up to 4 positions), one dispatch over all their row tiles */
typedef struct { const mat *w; uint32_t x0; int xs; uint32_t y0; int ys; } mmp;
static void mm_jobs(const mmp *M, int n, int b0, int nb) {
	if (njobs + n > JOBCAP || ncnt + 1 > CNTCAP) { fprintf(stderr, "air: job buffer full\n"); exit(1); }
	job *g = (job *)bJobs.contents + njobs; uint32_t tiles = 0; int big = 0, kmax = 0;
	for (int i = 0; i < n; i++) {
		const mat *w = M[i].w; uint32_t x[4] = { 0 };
		for (int k = 0; k < nb; k++) x[k] = M[i].x0 + (b0 + k) * M[i].xs;
		g[i] = (job){ (uint32_t)w->off, (uint32_t)(w->off >> 32), w->N, w->K, w->gs, nb, x[0], x[1], x[2], x[3],
			M[i].y0 + b0 * M[i].ys, M[i].ys, tiles, {0}, !w->nl, 0, 4 };
		memcpy(g[i].tbl, w->nl ? NL : E2, 16);
		tiles += (w->N / 4 + 31) / 32;
		if (nb * (int)w->K > 14336) big = 1;
		if ((int)w->K > kmax) kmax = w->K;
	}
	((uint32_t *)bCnt.contents)[ncnt] = n;
	set(big ? P_mmbig[nb] : P_mm[nb]);
	if (!big) [E setThreadgroupMemoryLength:((size_t)nb * kmax * 2 + 15) & ~(size_t)15 atIndex:0];
	arg(0, bW); arg(1, bS); arg(2, bS);
	[E setBuffer:bJobs offset:sizeof(job) * njobs atIndex:3]; [E setBuffer:bCnt offset:4 * ncnt atIndex:4];
	grid(tiles, 256);
	njobs += n; ncnt++;
}
/* prefill: simdgroup-matrix GEMM, 64 rows x 64 tokens per threadgroup */
static void gemm(const mat *w, uint32_t x, int xs, uint32_t y, int ys, int T) {
	gjob j = { (uint32_t)w->off, (uint32_t)(w->off >> 32), w->N, w->K, w->gs, !w->nl, {0}, x, xs, y, ys, T, 0, 0, 0 };
	memcpy(j.tbl, w->nl ? NL : E2, 16);
	set(P_gemm); arg(0, bW); arg(1, bS); P(2, j); grid2((w->N + 63) / 64, (T + 63) / 64, 128);
}
static int pf;	/* encoding a prefill */
static void mm(const mmp *M, int n, int B) {
	if (pf) { for (int i = 0; i < n; i++) gemm(M[i].w, M[i].x0, M[i].xs, M[i].y0, M[i].ys, B); return; }
	for (int b0 = 0; b0 < B; b0 += 4) mm_jobs(M, n, b0, B - b0 < 4 ? B - b0 : 4);
}
static void mm1(const mat *w, uint32_t x, int xs, uint32_t y, int ys, int B) { mmp p = { w, x, xs, y, ys }; mm(&p, 1, B); }

static cmat cm(const mat *w) { cmat c = { (uint32_t)w->off, (uint32_t)(w->off >> 32), w->gs, !w->nl }; memcpy(c.tbl, w->nl ? NL : E2, 16); return c; }
/* loop controllers: c = (1 - tanh(F h)) c + tanh(G h) m, h = D [rms c; rms m], for J modules per position */
static void ctl(const layer *L, int B, int J, const int *mod, uint32_t cur, int cbs, int cjs, uint32_t msg, int mbs, int mjs) {
	if (B <= 4) {
		struct { uint32_t cur, cbs, cjs, msg, mbs, mjs, J, hr; cmat mt[3]; } a = { cur, cbs, cjs, msg, mbs, mjs, J, Y->hr };
		struct { uint32_t cur, cbs, cjs, msg, mbs, mjs, J, hr, B; cmat mt[6]; } b = { cur, cbs, cjs, msg, mbs, mjs, J, Y->hr, B };
		for (int j = 0; j < J; j++) { a.mt[j] = cm(&L->ld[mod[j]]); b.mt[2 * j] = cm(&L->lg[mod[j]]); b.mt[2 * j + 1] = cm(&L->lf[mod[j]]); }
		set(P_ctl_a); arg(0, bS); arg(1, bW); P(2, a); grid(B * J, 256);
		set(P_ctl_b); arg(0, bS); arg(1, bW); P(2, b); grid(J * 9, 256);
		return;
	}
	struct { uint32_t cur, cbs, cjs, msg, mbs, mjs, h2, g, f, J; } p = { cur, cbs, cjs, msg, mbs, mjs, Y->h2, Y->cg, Y->cf, J };
	set(P_ctl_prep); arg(0, bS); P(1, p); grid(B * J, 256);
	mmp a[3], gf[6];
	for (int j = 0; j < J; j++) a[j] = (mmp){ &L->ld[mod[j]], Y->h2 + j * 2 * NE, J * 2 * NE, Y->hr + j * RANK, J * RANK };
	mm(a, J, B);
	for (int j = 0; j < J; j++) {
		gf[2 * j] = (mmp){ &L->lg[mod[j]], Y->hr + j * RANK, J * RANK, Y->cg + j * NE, J * NE };
		gf[2 * j + 1] = (mmp){ &L->lf[mod[j]], Y->hr + j * RANK, J * RANK, Y->cf + j * NE, J * NE };
	}
	mm(gf, 2 * J, B);
	set(P_ctl_combine); arg(0, bS); P(1, p); threads(B * J * NE);
}

/* decode attention: nq query rows (3 per kv head per position) against keys [k0, kend]; qpos = position of the first query row
   (causal) or of the whole block (non-causal); window masking and chunk 256 on ring visits */
static void attn(id<MTLBuffer> K, id<MTLBuffer> V, id<MTLBuffer> K8, id<MTLBuffer> V8, id<MTLBuffer> KS, int nt, int vsoff, int capv, int swa,
                 int causal, int bias, uint32_t relb, uint32_t q, uint32_t r, int nq, int qpos, int k0, int kend, uint32_t sink, uint32_t gate, uint32_t out, int npos) {
	int ch = swa ? 256 : ATTCH, nch = (kend - k0 + ch) / ch;
	if (nch > ATTCH) { fprintf(stderr, "air: context too long\n"); exit(1); }
	if (K8) {
		struct { uint32_t nq, pos0, k0, kend, nch, q, r, relb, pm, pl, pacc, bias, nt, vs; } p =
			{ nq, qpos, k0, kend, nch, q, r, relb, Y->pm, Y->pl, Y->pacc, bias, nt, vsoff };
		set(P_attn8); arg(0, K8); arg(1, V8); arg(2, bS); arg(3, bM); P(4, p); arg(5, KS);
	} else {
		struct { uint32_t nq, pos0, k0, kend, cap, ring, swa, nch, q, r, relb, pm, pl, pacc, causal, bias; } p =
			{ nq, qpos, k0, kend, capv, swa, swa, nch, q, r, relb, Y->pm, Y->pl, Y->pacc, causal, bias };
		set(P_attn); arg(0, K); arg(1, V); arg(2, bS); arg(3, bM); P(4, p);
	}
	grid(NKV * nch, 256);
	struct { uint32_t nch, pm, pl, pacc, sink, gate, att, hass, hasg; } m = { nch, Y->pm, Y->pl, Y->pacc, sink, gate, out, sink != 0, gate != 0 };
	set(P_attn_merge); arg(0, bS); P(1, m); grid(npos * NH, HD);
}

/* ---------------- drafter ---------------- */
static void hrope(uint32_t src, int ss, int nheads, int normw, int rope, int pos0, id<MTLBuffer> C, int nvec, int l, int kind8) {
	id<MTLBuffer> Q8 = kind8 && l >= 0 ? (kind8 == 1 ? bDK8[l] : bDV8[l]) : nil;
	if (!Q8) kind8 = 0;
	struct { uint32_t src, ss, nh, nw, hasn, rope, invf, pos0, store, kind8, nt, vs; } p =
		{ src, ss, nheads, normw < 0 ? 0 : normw, normw >= 0, rope, invf, pos0, C != nil, kind8, dntile, dcap * NKV };
	set(P_headnorm_rope); arg(0, bS); arg(1, bM); arg(2, C ? C : bS); P(3, p); arg(4, Q8 ? Q8 : bS); arg(5, kind8 ? bDKS[l] : bS);
	grid(nvec * nheads, HD);
}
static void add(uint32_t x, uint32_t y, int n) { struct { uint32_t x, y, n; } p = { x, y, n }; set(P_add); arg(0, bS); P(1, p); threads(n); }

/* target features feat[n][4][NE] -> drafter context K/V at pos0.., markov context ring */
static void inject(uint32_t feat, int pos0, int n) {
	for (int b0 = 0; b0 < n; b0 += DBLOCK) {
		int nb = n - b0 < DBLOCK ? n - b0 : DBLOCK, p0 = pos0 + b0;
		struct { uint32_t feat, out, ors, fuse, n0, n1, n2, n3, pos0, ctxr; } p =
			{ feat + b0 * 4 * NE, DC, MAXB * NE, W->fuse, W->D[0].dfl, W->D[1].dfl, W->D[2].dfl, W->dnorm, p0, CTXR };
		set(P_dfuse); arg(0, bS); arg(1, bM); P(2, p); grid(nb * 4, 256);
		for (int l = 0; l < DL; l++) {
			mmp m[2] = { { &W->D[l].kc, DC + l * MAXB * NE, NE, DKK, KVD }, { &W->D[l].vc, DC + l * MAXB * NE, NE, DVV, KVD } };
			mm(m, 2, nb);
			hrope(DKK, KVD, NKV, W->D[l].kcn, 1, p0, bDK[l], nb, l, 1);
			hrope(DVV, KVD, NKV, -1, 0, p0, bDV[l], nb, l, 2);
		}
	}
}
/* one block: anchor row + mask rows through the 3 drafter layers; base logits of the drafted rows into lg */
static void draft(int pos0, int nd) {
	const int NR = nd + 1 > 4 ? nd + 1 : 4;
	for (int i = 1; i < NR; i++) memcpy(S + DX + i * NE, SM + W->mask, NE * 4);
	for (int l = 0; l < DL; l++) {
		const dlayer *D = &W->D[l];
		const int last = l == DL - 1 && nd < NR - 1, r0 = last ? 1 : 0, nr = last ? nd : NR;
		norm(DX, DH, D->an, NR, NE, NE);
		mmp kv[2] = { { &D->k, DH, NE, DKK, KVD }, { &D->v, DH, NE, DVV, KVD } };
		mm(kv, 2, NR); mm1(&D->q, DH + r0 * NE, NE, DQ + r0 * QD, QD, nr);
		hrope(DQ + r0 * QD, QD, NH, D->qn, 1, pos0 + r0, nil, nr, -1, 0);
		hrope(DKK, KVD, NKV, D->kn, 1, pos0, bDK[l], NR, l, 1);
		hrope(DVV, KVD, NKV, -1, 0, pos0, bDV[l], NR, l, 2);
		/* non-causal block: every row attends to the context and the whole block */
		attn(bDK[l], bDV[l], bDK8[l], bDV8[l], bDKS[l], dntile, dcap * NKV, 0, 0, 0, 0, 0, DQ + r0 * QD, 0, 3 * nr,
		     pos0 + NR - 1, D->swa && pos0 > SWA - 1 ? pos0 - (SWA - 1) : 0, pos0 + NR - 1, 0, 0, DATT + r0 * QD, nr);
		mm1(&D->o, DATT + r0 * QD, QD, DO + r0 * NE, NE, nr);
		add(DX + r0 * NE, DO + r0 * NE, nr * NE);
		norm(DX + r0 * NE, DH + r0 * NE, D->fn, nr, NE, NE);
		mmp gu[2] = { { &D->gate, DH + r0 * NE, NE, DG + r0 * DFF, DFF }, { &D->up, DH + r0 * NE, NE, DU + r0 * DFF, DFF } };
		mm(gu, 2, nr);
		struct { uint32_t g, u, n; } sw = { DG + r0 * DFF, DU + r0 * DFF, nr * DFF };
		set(P_swiglu); arg(0, bS); P(1, sw); threads(nr * DFF);
		mm1(&D->down, DG + r0 * DFF, DFF, DO + r0 * NE, NE, nr);
		add(DX + r0 * NE, DO + r0 * NE, nr * NE);
	}
	norm(DX + NE, DH + NE, W->doutn, nd, NE, NE);
	mm1(&W->out, DH + NE, NE, LD.lg, NVOCAB + 3, nd);
}

/* ---------------- target forward ---------------- */
static void embed(const int *tok, int B, uint32_t dst) {
	for (int b = 0; b < B; b++) {
		float *c = S + dst + b * NE; q8row(W->emb, tok[b], c);
		double ss = 0; for (int i = 0; i < NE; i++) ss += (double)c[i] * c[i];
		float r = 1.0f / sqrtf((float)(ss / NE) + EPS);
		for (int i = 0; i < NE; i++) c[i] *= r * SM[W->enorm + i];
	}
}
static const float *xlast;
static void finish(id<MTLCommandBuffer> cb) {
	[E endEncoding]; [cb commit]; [cb waitUntilCompleted];
	if (cb.error) fprintf(stderr, "air: gpu: %s\n", cb.error.description.UTF8String);
}

static void forward(const int *tok, int B, int pos0, int last) {
	@autoreleasepool {
	embed(tok, B, Y->cur);
	njobs = ncnt = 0;
	id<MTLCommandBuffer> cb = [queue commandBufferWithUnretainedReferences]; E = [cb computeCommandEncoder];
	int nb = 0, pre = 0;
	for (int v = 0; v < NVIS; v++) {
		const int ip = phys(v), swa = W->swa[v]; const layer *L = &W->L[ip];
		int msg = 0;
		if (ip >= CORE0 && ip < CORE1) {	/* looped core: save / restore the block list around each pass, cross-pass messages */
			int pass = (v - CORE0) / (CORE1 - CORE0);
			if (ip == CORE0 && pass == 0) { pre = nb; if (nb) copy(Y->blk, Y->preb, nb * NE, B, BLKS, BLKS); copy(Y->cur, Y->prec, NE, B, NE, NE); }
			else if (ip == CORE0) {
				int nbank = nb - (pre + 1);
				if (nbank > 0) copy(Y->blk + (pre + 1) * NE, Y->bank, nbank * NE, B, BLKS, BLKS);
				copy(Y->cur, Y->bankl, NE, B, NE, NE);
				if (pre) copy(Y->preb, Y->blk, pre * NE, B, BLKS, BLKS);
				copy(Y->prec, Y->cur, NE, B, NE, NE); nb = pre;
			}
			if (pass > 0) {
				int row = 2 * (ip - CORE0), nbk = (CORE1 - CORE0) / BLOCK - 1;
				msg = 1;
				mix(nbk, Y->bank, Y->bankl, W->loopq + row * NE, 2, Y->msg, 4 * NE, 3 * NE, B, -1, -1, -1);
				mix(nbk, Y->bank, Y->bankl, W->loopkvq + row * NE, 2, Y->msg + NE, 4 * NE, NE, B, -1, -1, -1);
			}
		}
		if (v == 5 || v == 23 || v == 41) copy(Y->cur, Y->feat + (v == 5 ? 0 : v == 23 ? 1 : 2) * NE, NE, B, NE, 4 * NE);
		/* attention inputs: depth mix (+ norm when no controller follows) */
		int normed = 0;
		if (L->narq && !msg) { mix(nb, Y->blk, Y->cur, L->arq, L->narq, Y->inp, 3 * NE, NE, B, swa ? L->anorm : L->qn, swa ? -1 : L->kn, swa ? -1 : L->vn); normed = 1; }
		else if (L->narq) mix(nb, Y->blk, Y->cur, L->arq, L->narq, Y->inp, 3 * NE, NE, B, -1, -1, -1);
		else copy(Y->cur, Y->inp, NE, B, NE, 3 * NE);
		if (msg) {
			if (!L->narq || swa) { copy(Y->inp, Y->inp + NE, NE, B, 3 * NE, 3 * NE); copy(Y->inp, Y->inp + 2 * NE, NE, B, 3 * NE, 3 * NE); }
			int mod[3] = { 0, 1, 2 }; ctl(L, B, 3, mod, Y->inp, 3 * NE, NE, Y->msg, 4 * NE, NE);
		}
		if (!normed) for (int j = 0; j < (swa && !msg ? 1 : 3); j++) norm(Y->inp + j * NE, Y->inp + j * NE, swa ? L->anorm : j == 0 ? L->qn : j == 1 ? L->kn : L->vn, B, 3 * NE, 3 * NE);
		uint32_t i0 = Y->inp, i1 = !swa || msg ? Y->inp + NE : i0, i2 = !swa || msg ? Y->inp + 2 * NE : i0;
		mmp pj[8] = { { &L->q, i0, 3 * NE, Y->q, QD }, { &L->k, i1, 3 * NE, Y->kk, KVD }, { &L->v, i2, 3 * NE, Y->vv, KVD },
			{ &L->rel, i0, 3 * NE, Y->r, NH * NREL }, { &L->gate, i0, 3 * NE, Y->gate, QD }, { &L->sink, i0, 3 * NE, Y->sink, NH },
			{ &L->kdyn, i1, 3 * NE, Y->dk, 32 }, { &L->vdyn, i2, 3 * NE, Y->dv, 32 } };
		mm(pj, swa ? 6 : 8, B);
		/* k/v short conv, clip / norm, cache store (fp16 and, full attention, int8) */
		{ int h8 = bK8[v] != nil;
		  struct { uint32_t pos0, B, cap, ring, offk, offv, wk, wv, dynk, dynv, hasdyn; float qk, eps; uint32_t srck, srcv, has8, nt, vs, noring; } p =
			{ pos0, B, cap[v], swa, 0, KVD, L->kconv, L->vconv, Y->dk, Y->dv, !swa, QK_CLIP, EPS, Y->kk, Y->vv, h8, ntile[v], cap[v] * NKV, pf };
		  set(P_conv_kv); arg(0, bS); arg(1, bM); arg(2, bR[v]); arg(3, bK[v]); arg(4, bV[v]); P(5, p);
		  arg(6, h8 ? bK8[v] : bS); arg(7, h8 ? bV8[v] : bS); arg(8, h8 ? bKS[v] : bS); grid(B * NKV * 2, HD);
		  if (pf) { int r0 = B >= 3 ? B - 3 : 0; ring(v, Y->kk + r0 * KVD, KVD, 0, KVD, pos0 + r0, B - r0); ring(v, Y->vv + r0 * KVD, KVD, KVD, KVD, pos0 + r0, B - r0); } }
		{ struct { uint32_t q, r, qclip, rclip, full; float qk, rc; uint32_t pos0; float alpha, floor_; } p =
			{ Y->q, Y->r, L->qclip, L->rclip, !swa, QK_CLIP, REL_CLIP, pos0, 0.1f, 8192.0f };
		  set(P_qr); arg(0, bS); arg(1, bM); P(2, p); grid(B * NH, HD); }
		if (pf) {
			struct { uint32_t T, pos0, cap, ring, swa, q, r, relb, sink, gate, att, bias; } p =
				{ B, pos0, cap[v], swa, swa, Y->q, Y->r, L->relb, Y->sink, Y->gate, Y->att, 1 };
			set(P_attn_prefill); arg(0, bK[v]); arg(1, bV[v]); arg(2, bS); arg(3, bM); P(4, p); grid2(NKV, (B + 15) / 16, 256);
		} else for (int g0 = 0; g0 < B; g0 += 4) {	/* query groups of <= 4 positions */
			int gb = B - g0 < 4 ? B - g0 : 4;
			int q0 = pos0 + g0;
			attn(bK[v], bV[v], bK8[v], bV8[v], bKS[v], ntile[v], cap[v] * NKV, cap[v], swa, 1, 1, L->relb, Y->q + g0 * QD, Y->r + g0 * NH * NREL,
			     3 * gb, q0, swa && q0 > SWA - 1 ? q0 - (SWA - 1) : 0, q0 + gb - 1, Y->sink + g0 * NH, Y->gate + g0 * QD, Y->att + g0 * QD, gb);
		}
		mm1(&L->o, Y->att, QD, Y->o, NE, B);
		/* attention post conv + residual (or block push), ffn input mix + norm */
		int push = ip % BLOCK == 0;
		{ struct { uint32_t pos0, off, w, cur, ffn, blk, bs, nb, push, src, frq, xm, route, nw, noring; } p =
			{ pos0, 2 * KVD, L->apc, Y->cur, Y->ffn, Y->blk, BLKS, nb, push, Y->o, L->frq, Y->xm, !msg, L->fnorm, pf };
		  set(P_ffn_front); arg(0, bS); arg(1, bM); arg(2, bR[v]); P(3, p); grid(B, 512);
		  if (pf) { int r0 = B >= 3 ? B - 3 : 0; ring(v, Y->o + r0 * NE, NE, 2 * KVD, NE, pos0 + r0, B - r0); } }
		if (push) nb++;
		if (msg) { int mod[1] = { 3 }; ctl(L, B, 1, mod, Y->xm, NE, 0, Y->msg + 3 * NE, 4 * NE, 0); norm(Y->xm, Y->xm, L->fnorm, B, NE, NE); }
		/* router + top-8 */
		if (pf && B > 128) { struct { uint32_t w, x, y, T; } p = { L->router, Y->xm, Y->log, B };
			set(P_router_gemm); arg(0, bS); arg(1, bM); P(2, p); grid2(NEXP / 64, (B + 31) / 32, 128); }
		else for (int b0 = 0; b0 < B; b0 += 16) gemv(L->router, NEXP, NE, Y->xm + b0 * NE, NE, Y->log + b0 * NEXP, NEXP, B - b0 < 16 ? B - b0 : 16);
		{ struct { uint32_t logit, bias, ids, wts; float ws; } p = { Y->log, L->ebias, Y->ids, Y->wts, WSCALE };
		  set(P_topk); arg(0, bS); arg(1, bM); P(2, p); grid(B, NEXP); }
		/* experts */
		if (pf) {	/* counting sort by expert, grouped GEMMs over token tiles */
			const int mt = B * NUSED / NEXP < 12 ? 8 : 32;
			struct { uint32_t B, ids, smap, stok, tiles, mt; } pp = { B, Y->ids, Y->smap, Y->stok, Y->tiles, mt };
			set(P_moe_plan_big); arg(0, bS); [E setBuffer:bArgs offset:4 * 3 * 2 * v atIndex:1]; P(2, pp); grid(1, 1024);
			for (int ph = 0; ph < 2; ph++) {
				if (ph) { struct { uint32_t gu, act, n; } s = { Y->gu, Y->act, B * NUSED * NFF }; set(P_situ_all); arg(0, bS); P(1, s); threads(B * NUSED * NFF); }
				struct { uint32_t layer, ph, xm, gu, dn, stok, tiles, act; } g = { ip, ph, Y->xm, Y->gu, Y->dn, Y->stok, Y->tiles, Y->act };
				set(mt == 8 ? P_gmoe8 : P_gmoe32); arg(0, bW); arg(1, bS); arg(2, bET); P(3, g);
				[E dispatchThreadgroupsWithIndirectBuffer:bArgs indirectBufferOffset:4 * 3 * (2 * v + ph) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
			}
		} else {	/* job list of (expert, <= 4 positions) built on the gpu */
			if (njobs + 2 * MAXJ > JOBCAP || ncnt + 9 > CNTCAP) { fprintf(stderr, "air: job buffer full\n"); exit(1); }
			struct { uint32_t B, layer, ids, xm, gu, act, dn, smap, jobs, counts, args, maxjobs; } p = { B, ip, Y->ids, Y->xm, Y->gu, Y->act, Y->dn, Y->smap, 0, 0, 0, MAXJ };
			set(P_moe_plan); arg(0, bS); arg(1, bET); [E setBuffer:bJobs offset:sizeof(job) * njobs atIndex:2];
			[E setBuffer:bCnt offset:4 * ncnt atIndex:3]; [E setBuffer:bArgs offset:4 * 3 * 2 * v atIndex:4]; P(5, p); grid(1, 128);
			for (int ph = 0; ph < 2; ph++) {
				set(P_mmmoe); [E setThreadgroupMemoryLength:(size_t)(B < 4 ? B : 4) * (ph ? NFF : NE) * 2 atIndex:0]; arg(0, bW); arg(1, bS); arg(2, bS);
				[E setBuffer:bJobs offset:sizeof(job) * (njobs + ph * MAXJ) atIndex:3]; [E setBuffer:bCnt offset:4 * (ncnt + ph) atIndex:4];
				[E dispatchThreadgroupsWithIndirectBuffer:bArgs indirectBufferOffset:4 * 3 * (2 * v + ph) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
			}
			njobs += 2 * MAXJ; ncnt += 9;
		}
		{ struct { uint32_t smap, wts, dn, out, B; } p = { Y->smap, Y->wts, Y->dn, Y->ym, B };
		  set(P_moe_combine); arg(0, bS); P(1, p); threads(B * NE); }
		{ struct { uint32_t pos0, off, w, cur, ffn, src, noring; } p = { pos0, 2 * KVD + NE, L->fpc, Y->cur, Y->ffn, Y->ym, pf };
		  set(P_post); arg(0, bS); arg(1, bM); arg(2, bR[v]); P(3, p); grid(B, 1024);
		  if (pf) { int r0 = B >= 3 ? B - 3 : 0; ring(v, Y->ym + r0 * NE, NE, 2 * KVD + NE, NE, pos0 + r0, B - r0); } }
	}
	/* output mix + norm, 4-bit lm head, top-16 candidates per row */
	mix(nb, Y->blk, Y->cur, W->oq, 1, Y->x, NE, 0, B, W->onorm, -1, -1);
	copy(Y->x, Y->feat + 3 * NE, NE, B, NE, 4 * NE);
	const int lm0 = last ? B - 1 : 0, nlm = B - lm0, lgr = pf ? 0 : lm0;
	{ int sv = pf; pf = 0; mm1(&W->out, Y->x + lm0 * NE, NE, Y->lg + lgr * (NVOCAB + 3), NVOCAB + 3, nlm); pf = sv; }
	{ struct { uint32_t lg, lgs, n, nch, part, K; } p = { Y->lg + lgr * (NVOCAB + 3), NVOCAB + 3, NVOCAB, NTOPCH, Y->part, TOPC };
	  set(P_top_a); arg(0, bS); P(1, p); grid(nlm * NTOPCH, 256);
	  set(P_top_b); arg(0, bS); [E setBuffer:bCand offset:4 * TOPC * lgr atIndex:1]; P(2, p); grid(nlm, 1024); }
	if (pf) inject(Y->feat, pos0, B);
	finish(cb);
	xlast = S + Y->x;
	}
}

void gpu_forward(const int *tok, int B, int pos0, int last, int *cand) {
	if (B > MAXB) { fprintf(stderr, "air: batch too large\n"); exit(1); }
	forward(tok, B, pos0, last);
	memcpy(cand, bCand.contents, sizeof(int) * B * TOPC);
}
void gpu_prefill(const int *tok, int T, int pos0, int *cand) {
	if (T > PMAX) { fprintf(stderr, "air: prefill chunk too large\n"); exit(1); }
	Y = &LP; pf = 1;
	forward(tok, T, pos0, 1);
	Y = &LD; pf = 0;
	memcpy(cand, bCand.contents, sizeof(int) * TOPC);
}
const float *gpu_x(void) { return xlast; }

/* inject the features of the last decode forward (ni rows at ipos), then draft nd tokens after anchor at pos0:
   per drafted row the top-DCAND base candidates and the out-normed hidden x (for the markov head) */
void gpu_draft(int ipos, int ni, int anchor, int pos0, int nd, int *cand, const float **x) {
	@autoreleasepool {
	njobs = ncnt = 0;
	id<MTLCommandBuffer> cb = [queue commandBufferWithUnretainedReferences]; E = [cb computeCommandEncoder];
	if (ni > 0) inject(LD.feat, ipos, ni);
	if (nd > 0) {
		float *x0 = S + DX; q8row(W->emb, anchor, x0);
		double ss = 0; for (int i = 0; i < NE; i++) ss += (double)x0[i] * x0[i];
		float r = 1.0f / sqrtf((float)(ss / NE) + EPS);
		for (int i = 0; i < NE; i++) x0[i] *= r * SM[W->enorm + i];
		draft(pos0, nd);
	}
	finish(cb);
	for (int i = 0; i < nd; i++) {	/* top-DCAND of the base logits (unified memory) */
		const float *l = S + LD.lg + (size_t)i * (NVOCAB + 3); float tv[DCAND]; int *c = cand + i * DCAND, n = 0;
		for (int v = 0; v < NVOCAB; v++) {
			float val = l[v];
			if (n == DCAND && val <= tv[n - 1]) continue;
			int q = n < DCAND ? n++ : DCAND - 1;
			while (q > 0 && tv[q - 1] < val) { tv[q] = tv[q - 1]; c[q] = c[q - 1]; q--; }
			tv[q] = val; c[q] = v;
		}
	}
	if (x) *x = S + DH + NE;
	}
}
float *gpu_ctxring(void) { return S + CTXR; }
float *gpu_conv(int v) { return bR[v].contents; }

/* state io: caches of the first n positions (ring visits: the whole ring), conv rings, drafter caches + context ring */
static int io(FILE *f, void *p, size_t n, int wr) { return n == 0 || (wr ? fwrite(p, 1, n, f) : fread(p, 1, n, f)) == n; }
static int io8(FILE *f, id<MTLBuffer> K8, id<MTLBuffer> V8, id<MTLBuffer> KS, int nt, int cp, int n, int wr) {
	uint8_t *k8 = K8.contents, *v8 = V8.contents; float *ks = KS.contents; int ok = 1, ntl = (n + 31) / 32;
	for (int h = 0; h < NKV; h++) {
		ok &= io(f, k8 + (size_t)h * nt * 32 * HD, (size_t)n * HD, wr);
		ok &= io(f, v8 + (size_t)h * nt * HD * 32, (size_t)ntl * HD * 32, wr);
	}
	return ok & io(f, ks, (size_t)n * NKV * 4, wr) & io(f, ks + (size_t)cp * NKV, (size_t)n * NKV * 4, wr);
}
int gpu_state(FILE *f, int n, int wr) {
	int ok = 1;
	for (int v = 0; v < NVIS; v++) {
		size_t rows = W->swa[v] ? cap[v] : n;
		ok &= io(f, bK[v].contents, rows * KVD * 2, wr) & io(f, bV[v].contents, rows * KVD * 2, wr) & io(f, bR[v].contents, HIST * CONVW * 4, wr);
		if (bK8[v]) ok &= io8(f, bK8[v], bV8[v], bKS[v], ntile[v], cap[v], n, wr);
	}
	for (int l = 0; l < DL; l++) {
		ok &= io(f, bDK[l].contents, (size_t)n * KVD * 2, wr) & io(f, bDV[l].contents, (size_t)n * KVD * 2, wr);
		if (bDK8[l]) ok &= io8(f, bDK8[l], bDV8[l], bDKS[l], dntile, dcap, n, wr);
	}
	return ok & io(f, S + CTXR, CTXRING * NE * 4, wr) ? 0 : -1;
}
