/* air.c - Inkling 2 Mini on Apple silicon: weight prep, tokenizer, speculative decoding, chat, Anthropic API server. */
#include <arm_neon.h>
#include <arpa/inet.h>
#include <ctype.h>
#include <dirent.h>
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <mach/mach_time.h>
#include <math.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>
#include "air.h"

wfile *W;
float *SM;
uint8_t *AR;
static uint8_t *Q8E, *Q8U, *VOC;
static uint16_t *W1;
static __fp16 *JNT;
static float *HOUT;
static uint32_t *VOFF;

static double now(void) {
	static mach_timebase_info_data_t tb;
	if (!tb.denom) mach_timebase_info(&tb);
	return (double)mach_absolute_time() * tb.numer / tb.denom * 1e-9;
}
static void die(const char *fmt, ...) {
	va_list ap; va_start(ap, fmt); fprintf(stderr, "air: "); vfprintf(stderr, fmt, ap); fprintf(stderr, "\n"); va_end(ap); exit(1);
}
static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) die("out of memory"); return p; }

/* ---------------- growable buffers ---------------- */
typedef struct { char *p; size_t n, cap; } str;
static void sadd(str *b, const char *s, size_t n) {
	if (b->n + n + 1 > b->cap) { b->cap = (b->n + n + 1) * 2; b->p = realloc(b->p, b->cap); }
	memcpy(b->p + b->n, s, n); b->n += n; b->p[b->n] = 0;
}
static void sputs(str *b, const char *s) { sadd(b, s, strlen(s)); }
static void sprint(str *b, const char *fmt, ...) {
	va_list ap; va_start(ap, fmt); int n = vsnprintf(NULL, 0, fmt, ap); va_end(ap);
	char *t = xmalloc(n + 1); va_start(ap, fmt); vsnprintf(t, n + 1, fmt, ap); va_end(ap); sadd(b, t, n); free(t);
}
static void sfree(str *b) { free(b->p); memset(b, 0, sizeof *b); }
static void sjson(str *b, const char *s, size_t n) {	/* json.dumps(ensure_ascii=False) */
	sadd(b, "\"", 1);
	size_t i0 = 0;
	for (size_t i = 0; i < n; i++) {
		unsigned char c = s[i]; const char *e = NULL; char u[8];
		if (c == '"') e = "\\\""; else if (c == '\\') e = "\\\\"; else if (c == '\n') e = "\\n"; else if (c == '\r') e = "\\r";
		else if (c == '\t') e = "\\t"; else if (c == '\b') e = "\\b"; else if (c == '\f') e = "\\f";
		else if (c < 0x20) { snprintf(u, sizeof u, "\\u%04x", c); e = u; }
		if (e) { sadd(b, s + i0, i - i0); sputs(b, e); i0 = i + 1; }
	}
	sadd(b, s + i0, n - i0); sadd(b, "\"", 1);
}
typedef struct { int *p, n, cap; } ivec;
static void ipush(ivec *v, int x) { if (v->n == v->cap) { v->cap = v->cap ? v->cap * 2 : 1024; v->p = realloc(v->p, sizeof(int) * v->cap); } v->p[v->n++] = x; }

/* ---------------- json ---------------- */
enum { JNULL, JFALSE, JTRUE, JNUM, JSTR, JARR, JOBJ };
typedef struct J { int t, n; char *s; size_t len; struct J **a; char **k; size_t *kl; } J;	/* JNUM keeps its text */
typedef struct { const char *p, *e; int err; } jp;
static void jws(jp *q) { while (q->p < q->e && isspace((unsigned char)*q->p)) q->p++; }
static void utf8(str *b, unsigned c) {
	char o[4]; int n;
	if (c < 0x80) { o[0] = c; n = 1; }
	else if (c < 0x800) { o[0] = 0xC0 | c >> 6; o[1] = 0x80 | (c & 63); n = 2; }
	else if (c < 0x10000) { o[0] = 0xE0 | c >> 12; o[1] = 0x80 | (c >> 6 & 63); o[2] = 0x80 | (c & 63); n = 3; }
	else { o[0] = 0xF0 | c >> 18; o[1] = 0x80 | (c >> 12 & 63); o[2] = 0x80 | (c >> 6 & 63); o[3] = 0x80 | (c & 63); n = 4; }
	sadd(b, o, n);
}
static int hex4(const char *p) {
	int v = 0;
	for (int i = 0; i < 4; i++) { int c = tolower(p[i]); if (!isxdigit(c)) return -1; v = v * 16 + (isdigit(c) ? c - '0' : c - 'a' + 10); }
	return v;
}
static int jstr(jp *q, char **out, size_t *len) {
	if (q->p >= q->e || *q->p != '"') return -1;
	q->p++; str b = { 0 }; sadd(&b, "", 0);
	while (q->p < q->e && *q->p != '"') {
		if (*q->p != '\\') { const char *s = q->p; while (q->p < q->e && *q->p != '"' && *q->p != '\\') q->p++; sadd(&b, s, q->p - s); continue; }
		if (++q->p >= q->e) break;
		char c = *q->p++;
		switch (c) {
		case 'n': sputs(&b, "\n"); break; case 't': sputs(&b, "\t"); break; case 'r': sputs(&b, "\r"); break;
		case 'b': sputs(&b, "\b"); break; case 'f': sputs(&b, "\f"); break;
		case 'u': {
			int cp = q->e - q->p >= 4 ? hex4(q->p) : -1; if (cp < 0) { sfree(&b); return -1; }
			q->p += 4;
			if (cp >= 0xD800 && cp < 0xDC00 && q->e - q->p >= 6 && q->p[0] == '\\' && q->p[1] == 'u') {
				int lo = hex4(q->p + 2); if (lo >= 0xDC00 && lo < 0xE000) { cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); q->p += 6; }
			}
			utf8(&b, cp); break;
		}
		default: sadd(&b, &c, 1);
		}
	}
	if (q->p >= q->e) { sfree(&b); return -1; }
	q->p++; *out = b.p; *len = b.n; return 0;
}
static J *jval(jp *q, int depth) {
	jws(q);
	if (q->p >= q->e || depth > 200) { q->err = 1; return NULL; }
	J *j = calloc(1, sizeof *j); char c = *q->p;
	if (c == '{' || c == '[') {
		int obj = c == '{', cap = 0; j->t = obj ? JOBJ : JARR; q->p++; jws(q);
		if (q->p < q->e && *q->p == (obj ? '}' : ']')) { q->p++; return j; }
		for (;;) {
			char *k = NULL; size_t kl = 0;
			if (obj) { jws(q); if (jstr(q, &k, &kl)) { q->err = 1; return j; } jws(q); if (q->p >= q->e || *q->p++ != ':') { q->err = 1; free(k); return j; } }
			J *v = jval(q, depth + 1);
			if (q->err) { free(k); return j; }
			if (j->n == cap) { cap = cap ? 2 * cap : 4; j->a = realloc(j->a, sizeof(J *) * cap); if (obj) { j->k = realloc(j->k, sizeof(char *) * cap); j->kl = realloc(j->kl, sizeof(size_t) * cap); } }
			if (obj) { j->k[j->n] = k; j->kl[j->n] = kl; }
			j->a[j->n++] = v; jws(q);
			if (q->p < q->e && *q->p == ',') { q->p++; continue; }
			if (q->p < q->e && *q->p == (obj ? '}' : ']')) { q->p++; return j; }
			q->err = 1; return j;
		}
	}
	if (c == '"') { j->t = JSTR; if (jstr(q, &j->s, &j->len)) q->err = 1; return j; }
	if (!strncmp(q->p, "true", 4)) { j->t = JTRUE; q->p += 4; return j; }
	if (!strncmp(q->p, "false", 5)) { j->t = JFALSE; q->p += 5; return j; }
	if (!strncmp(q->p, "null", 4)) { j->t = JNULL; q->p += 4; return j; }
	const char *s = q->p; while (q->p < q->e && (isdigit((unsigned char)*q->p) || strchr("+-.eE", *q->p))) q->p++;
	if (q->p == s) q->err = 1;
	j->t = JNUM; j->len = q->p - s; j->s = strndup(s, j->len); return j;
}
static void jfree(J *j) {
	if (!j) return;
	for (int i = 0; i < j->n; i++) { jfree(j->a[i]); if (j->k) free(j->k[i]); }
	free(j->a); free(j->k); free(j->kl); free(j->s); free(j);
}
static J *jparse(const char *s, size_t n) {
	jp q = { s, s + n, 0 }; J *j = jval(&q, 0); jws(&q);
	if (q.err || q.p != q.e) { jfree(j); return NULL; }
	return j;
}
static J *jget(const J *o, const char *k) {
	if (o && o->t == JOBJ) for (int i = 0; i < o->n; i++) if (!strcmp(o->k[i], k)) return o->a[i];
	return NULL;
}
static const char *jsv(const J *o, const char *k) { J *v = jget(o, k); return v && v->t == JSTR ? v->s : NULL; }
static double jnum(const J *o, const char *k, double d) { J *v = jget(o, k); return v && v->t == JNUM ? atof(v->s) : d; }
static int *jorder(const J *o, int sort) {		/* key order: as given, or sorted */
	int *ord = xmalloc(sizeof(int) * (o->n + 1));
	for (int i = 0; i < o->n; i++) ord[i] = i;
	for (int i = 1; sort && i < o->n; i++) { int x = ord[i], k = i; while (k > 0 && strcmp(o->k[ord[k - 1]], o->k[x]) > 0) { ord[k] = ord[k - 1]; k--; } ord[k] = x; }
	return ord;
}
static void jdump(str *b, const J *j, int sort) {	/* compact separators; sort = recursive sort_keys */
	switch (j->t) {
	case JNULL: sputs(b, "null"); break;
	case JFALSE: sputs(b, "false"); break;
	case JTRUE: sputs(b, "true"); break;
	case JNUM: sadd(b, j->s, j->len); break;
	case JSTR: sjson(b, j->s, j->len); break;
	case JARR: sputs(b, "["); for (int i = 0; i < j->n; i++) { if (i) sputs(b, ","); jdump(b, j->a[i], sort); } sputs(b, "]"); break;
	case JOBJ: {
		int *o = jorder(j, sort); sputs(b, "{");
		for (int i = 0; i < j->n; i++) { if (i) sputs(b, ","); sjson(b, j->k[o[i]], j->kl[o[i]]); sputs(b, ":"); jdump(b, j->a[o[i]], sort); }
		sputs(b, "}"); free(o);
	}
	}
}

/* ---------------- numerics ---------------- */
static float bf16(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static float dot(const float *a, const float *b, int n) {
	float32x4_t s0 = vdupq_n_f32(0), s1 = s0;
	for (int i = 0; i < n; i += 8) { s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i)); s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4)); }
	return vaddvq_f32(vaddq_f32(s0, s1));
}
void q8row(uint64_t off, int row, float *out) {
	const uint8_t *r = (off == W->emb ? Q8E : Q8U) + (size_t)row * Q8ROW;
	const int8_t *q = (const int8_t *)r; const __fp16 *s = (const __fp16 *)(r + NE);
	for (int b = 0; b < NE / 32; b++) { float d = s[b]; for (int i = 0; i < 32; i++) out[b * 32 + i] = q[b * 32 + i] * d; }
}
static float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

/* ================= prep: safetensors (bf16 + NVFP4 experts) -> one packed file ================= */
static const int8_t CB_NL[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };
static float e5m3(int c) { return c ? ldexpf(1.0f + (c & 7) / 8.0f, (c >> 3) - 10) : 0.0f; }
static int e5m3_code(float v) {		/* nearest E5M3 code, v >= 0 */
	if (!(v > 0)) return 0;
	int e; float m = frexpf(v, &e); int ex = e - 1, man = (int)lrintf((2 * m - 1) * 8);
	if (man == 8) { man = 0; ex++; }
	int c5 = ex + 10;
	if (c5 < 0) return 0;
	if (c5 > 31) { c5 = 31; man = 7; }
	int c = c5 << 3 | man;
	return c ? c : 1;
}
static float e4m3(uint8_t x) { int ex = x >> 3 & 15, m = x & 7; return ex ? ldexpf(1 + m / 8.0f, ex - 7) : ldexpf(m, -9); }
/* nibble for (row rr of the group, k kk of the 16-block) */
static void put4(uint8_t *rec, int rr, int kk, int code) {
	int j = kk >> 2, b = (j >> 1) * 16 + rr * 4 + (kk & 3);
	rec[b] = j & 1 ? (uint8_t)((rec[b] & 0x0F) | code << 4) : (uint8_t)((rec[b] & 0xF0) | code);
}
#define REC(q, g, kb, KB) ((q) + (size_t)(g) * (KB) * 36 + (size_t)((kb) >> 2) * 144 + ((kb) & 3) * 32)
#define SCL(q, g, kb, KB) ((q) + (size_t)(g) * (KB) * 36 + (size_t)((kb) >> 2) * 144 + 128 + ((kb) & 3) * 4)

typedef struct { const uint8_t *p; char dt[8]; int64_t sh[4]; int nd; } tens;
typedef struct { J *h; const uint8_t *base; } stfile;
static stfile SF[16]; static int nsf;
static void st_open(const char *path) {
	int fd = open(path, O_RDONLY); struct stat st;
	if (fd < 0 || fstat(fd, &st)) die("cannot open %s", path);
	const uint8_t *m = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	if (m == MAP_FAILED) die("cannot map %s", path);
	uint64_t n; memcpy(&n, m, 8);
	J *h = jparse((const char *)m + 8, n);
	if (!h) die("bad safetensors header in %s", path);
	SF[nsf++] = (stfile){ h, m + 8 + n };
}
static tens T(const char *fmt, ...) {
	char name[256]; va_list ap; va_start(ap, fmt); vsnprintf(name, sizeof name, fmt, ap); va_end(ap);
	for (int f = 0; f < nsf; f++) {
		J *t = jget(SF[f].h, name);
		if (!t) continue;
		tens r = { 0 }; J *sh = jget(t, "shape"), *off = jget(t, "data_offsets");
		snprintf(r.dt, sizeof r.dt, "%s", jsv(t, "dtype"));
		r.nd = sh->n; for (int i = 0; i < sh->n; i++) r.sh[i] = atoll(sh->a[i]->s);
		r.p = SF[f].base + atoll(off->a[0]->s);
		return r;
	}
	die("missing tensor %s", name); return (tens){ 0 };
}
static int has(const char *name) { for (int f = 0; f < nsf; f++) if (jget(SF[f].h, name)) return 1; return 0; }
static float tf(tens t, size_t i) { return !strcmp(t.dt, "F32") ? ((const float *)t.p)[i] : bf16(((const uint16_t *)t.p)[i]); }

static int ofd; static uint64_t apos, fpos, A0;	/* arena offsets are relative to A0 */
static float *sm; static size_t nsm, capsm;
static uint32_t small(tens t, size_t n) {		/* append n fp32 values to the smalls */
	if (nsm + n > capsm) { capsm = (nsm + n) * 2; sm = realloc(sm, capsm * 4); }
	for (size_t i = 0; i < n; i++) sm[nsm + i] = tf(t, i);
	uint32_t o = (uint32_t)nsm; nsm += (n + 3) & ~(size_t)3; return o;
}
static uint32_t conv(tens t, int ch, int stat) {	/* conv taps -> [ch][4]; static_w [4][ch] is stored flipped */
	tens c = t; float *v = xmalloc(sizeof(float) * ch * 4);
	for (int i = 0; i < ch; i++) for (int j = 0; j < 4; j++) v[i * 4 + j] = stat ? tf(t, (size_t)(3 - j) * ch + i) : tf(t, (size_t)i * 4 + j);
	snprintf(c.dt, sizeof c.dt, "F32"); c.p = (const uint8_t *)v;
	uint32_t o = small(c, ch * 4); free(v); return o;
}
static void pw(const void *p, size_t n, uint64_t off) {
	for (size_t d = 0; d < n; ) { ssize_t k = pwrite(ofd, (const char *)p + d, n - d, off + d); if (k <= 0) die("write failed"); d += k; }
}
static mat newmat(int N, int K, int nl, float gs) {
	mat m = { apos, N, K, nl, gs };
	apos += ((size_t)N * K * 9 / 16 + 16383) & ~(size_t)16383; return m;
}
/* rows -> 4-bit, IQ4_NL codebook, per 16-block E5M3 scale searched for least squares error */
typedef void (^rowfn)(size_t r, float *x);
static mat quantf(int N, int K, rowfn get) {
	float *rmax = xmalloc(sizeof(float) * N), amax = 0;
	dispatch_apply(N, dispatch_get_global_queue(0, 0), ^(size_t r) { float x[K], m = 0; get(r, x); for (int i = 0; i < K; i++) m = fmaxf(m, fabsf(x[i])); rmax[r] = m; });
	for (int r = 0; r < N; r++) amax = fmaxf(amax, rmax[r]);
	free(rmax);
	mat m = newmat(N, K, 1, amax > 0 ? ldexpf(1, (int)ceilf(log2f(amax / 100)) - 18) : 1);
	const int KB = K / 16; uint8_t *q = calloc((size_t)N * K * 9 / 16, 1);
	static float mid[15]; for (int i = 0; i < 15; i++) mid[i] = 0.5f * (CB_NL[i] + CB_NL[i + 1]);
	dispatch_apply(N / 4, dispatch_get_global_queue(0, 0), ^(size_t g) {
		float row[K];
		for (int rr = 0; rr < 4; rr++) { get(g * 4 + rr, row); for (int kb = 0; kb < KB; kb++) {
			const float *x = row + kb * 16; float am = 0, vm = 0;
			for (int i = 0; i < 16; i++) if (fabsf(x[i]) > am) { am = fabsf(x[i]); vm = x[i]; }
			uint8_t *rec = REC(q, g, kb, KB);
			if (am == 0) { for (int i = 0; i < 16; i++) put4(rec, rr, i, 8); continue; }
			int bc = 0, bcodes[16], codes[16]; float be = INFINITY;
			for (int t2 = -8; t2 <= 8; t2++) {
				float d0 = fabsf(vm / (vm < 0 ? -127.0f + t2 : 113.0f + t2));
				for (int pass = 0; pass < 2; pass++) {	/* candidate scale, then one least-squares refit */
					int c = e5m3_code(d0 / m.gs); float d = e5m3(c) * m.gs;
					if (d == 0) break;
					float err = 0; double sxq = 0, sqq = 0;
					for (int i = 0; i < 16; i++) {
						float v = x[i] / d; int bi = 0;
						for (int st = 8; st; st >>= 1) if (bi + st < 16 && v > mid[bi + st - 1]) bi += st;
						codes[i] = bi; float dv = x[i] - CB_NL[bi] * d;
						err += dv * dv; sxq += x[i] * CB_NL[bi]; sqq += (double)CB_NL[bi] * CB_NL[bi];
					}
					if (err < be) { be = err; bc = c; memcpy(bcodes, codes, sizeof codes); }
					if (sqq <= 0 || (d0 = (float)(sxq / sqq)) < 0) break;
				}
			}
			SCL(q, g, kb, KB)[rr] = bc;
			for (int i = 0; i < 16; i++) put4(rec, rr, i, bcodes[i]);
		} }
	});
	pw(q, (size_t)N * K * 9 / 16, A0 + m.off); free(q);
	return m;
}
static mat quant(tens t, int N, int K) {
	const uint16_t *src = (const uint16_t *)t.p;
	return quantf(N, K, ^(size_t r, float *x) { for (int k = 0; k < K; k++) x[k] = bf16(src[r * K + k]); });
}
static int interleave(int r, int N, int w13) { return w13 ? (r < N / 2 ? 2 * r : 2 * (r - N / 2) + 1) : r; }
/* MXFP8 experts (E4M3 values, power-of-two scale per 32) are requantized */
static mat mxfp8(tens w, tens s, int e, int N, int K, int w13) {
	const uint8_t *pw8 = w.p + (size_t)e * N * K, *ps = s.p + (size_t)e * N * (K / 32);
	return quantf(N, K, ^(size_t r, float *x) { int o = interleave(r, N, w13);
		for (int k = 0; k < K; k++) { uint8_t b = pw8[(size_t)o * K + k]; float v = e4m3(b & 127) * ldexpf(1, ps[(size_t)o * (K / 32) + k / 32] - 127); x[k] = b & 128 ? -v : v; } });
}
/* NVFP4 experts (E2M1 nibbles, E4M3 scale per 16) repack losslessly; w13 interleaves gate and up rows */
static mat nvfp4(tens w, tens s, int e, int N, int K, int w13, float gs) {
	mat m = newmat(N, K, 0, 0.5f * gs);
	const int KB = K / 16; uint8_t *q = calloc((size_t)N * K * 9 / 16, 1);
	const uint8_t *pw4 = w.p + (size_t)e * N * K / 2, *ps = s.p + (size_t)e * N * KB;
	dispatch_apply(N / 4, dispatch_get_global_queue(0, 0), ^(size_t g) {
		for (int rr = 0; rr < 4; rr++) {
			int r = g * 4 + rr, o = interleave(r, N, w13);
			for (int kb = 0; kb < KB; kb++) {
				uint8_t sb = ps[(size_t)o * KB + kb]; float d = e4m3(sb); int c = e5m3_code(d);
				if (d != e5m3(c)) die("nvfp4 scale not representable");
				SCL(q, g, kb, KB)[rr] = c;
				uint8_t *rec = REC(q, g, kb, KB);
				for (int i = 0; i < 16; i++) { uint8_t b = pw4[(size_t)o * K / 2 + kb * 8 + i / 2]; put4(rec, rr, i, d == 0 ? 0 : i & 1 ? b >> 4 : b & 15); }
			}
		}
	});
	pw(q, (size_t)N * K * 9 / 16, A0 + m.off); free(q);
	return m;
}
static uint64_t q8tab(tens t, int rows) {		/* rows x NE -> int8 + fp16 scale per 32 */
	uint8_t *o = xmalloc((size_t)rows * Q8ROW); const uint16_t *src = (const uint16_t *)t.p;
	dispatch_apply(rows, dispatch_get_global_queue(0, 0), ^(size_t r) {
		int8_t *q = (int8_t *)(o + r * Q8ROW); __fp16 *s = (__fp16 *)(o + r * Q8ROW + NE);
		for (int b = 0; b < NE / 32; b++) {
			float am = 0; for (int i = 0; i < 32; i++) am = fmaxf(am, fabsf(bf16(src[r * NE + b * 32 + i])));
			__fp16 d = (__fp16)(am / 127); float id = (float)d > 0 ? 1 / (float)d : 0; s[b] = d;
			for (int i = 0; i < 32; i++) q[b * 32 + i] = (int8_t)lrintf(bf16(src[r * NE + b * 32 + i]) * id);
		}
	});
	uint64_t off = fpos; pw(o, (size_t)rows * Q8ROW, off); fpos += ((size_t)rows * Q8ROW + 16383) & ~(size_t)16383; free(o);
	return off;
}
static uint64_t blob(const void *p, size_t n) { uint64_t off = fpos; pw(p, n, off); fpos += (n + 16383) & ~(size_t)16383; return off; }
static void b64(const char *s, str *out) {
	static const char *A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	unsigned v = 0; int bits = 0;
	for (; *s && *s != '='; s++) { const char *c = strchr(A, *s); if (!c) break; v = v << 6 | (unsigned)(c - A); bits += 6;
		if (bits >= 8) { bits -= 8; char ch = (char)(v >> bits & 255); sadd(out, &ch, 1); } }
}

static int prep(const char *dir, const char *out) {
	char p[1024]; double t0 = now();
	DIR *d = opendir(dir); struct dirent *de;
	if (!d) die("cannot open %s", dir);
	while ((de = readdir(d)) && nsf < 15) {
		size_t l = strlen(de->d_name);
		if (l > 12 && !strcmp(de->d_name + l - 12, ".safetensors")) { snprintf(p, sizeof p, "%s/%s", dir, de->d_name); st_open(p); }
	}
	closedir(d);
	if (!has("model.llm.embed.weight") || !has("mask_embed")) die("%s: expected the Inkling 2 Mini checkpoint and mtp.safetensors", dir);
	ofd = open(out, O_RDWR | O_CREAT | O_TRUNC, 0644);
	if (ofd < 0) die("cannot create %s", out);
	static wfile w;
	memcpy(w.magic, AIRMAGIC, 8);
	A0 = (sizeof w + 16383) & ~(size_t)16383;
	for (int v = 0; v < NVIS; v++) { int ip = v < CORE1 ? v : v >= CORE0 + NPASS * (CORE1 - CORE0) ? v - (NPASS - 1) * (CORE1 - CORE0) : CORE0 + (v - CORE0) % (CORE1 - CORE0);
		w.swa[v] = ip % BLOCK != 5; }
	static const char *mods[4] = { "attn", "k", "v", "mlp" };
	for (int i = 0; i < NPHYS; i++) {
		layer *L = &w.L[i]; int swa = i % BLOCK != 5;
#define LT(s) T("model.llm.layers.%d." s, i)
		L->q = quant(LT("attn.wq_du.weight"), QD, NE); L->k = quant(LT("attn.wk_dv.weight"), KVD, NE); L->v = quant(LT("attn.wv_dv.weight"), KVD, NE);
		L->gate = quant(LT("attn.gate_attn.weight"), QD, NE); L->o = quant(LT("attn.wo_ud.weight"), NE, QD);
		L->rel = quant(LT("attn.wr_du.weight"), NH * NREL, NE); L->sink = quant(LT("attn.attn_sink_proj.weight"), NH, NE);
		if (!swa) { L->kdyn = quant(LT("attn.k_sconv.weight_proj"), 32, NE); L->vdyn = quant(LT("attn.v_sconv.weight_proj"), 32, NE); }
		if (i >= CORE0 && i < CORE1) for (int j = 0; j < 4; j++) {
			L->ld[j] = quant(T("model.llm.layers.%d.controllers.%s.down_weight", i, mods[j]), RANK, 2 * NE);
			L->lg[j] = quant(T("model.llm.layers.%d.controllers.%s.gain_weight", i, mods[j]), NE, RANK);
			L->lf[j] = quant(T("model.llm.layers.%d.controllers.%s.forget_weight", i, mods[j]), NE, RANK);
		}
		tens g13 = LT("mlp.experts.w13_weight"), s13 = LT("mlp.experts.w13_weight_scale"), g2 = LT("mlp.experts.w2_weight"), s2 = LT("mlp.experts.w2_weight_scale");
		if (!strcmp(g13.dt, "U8")) {		/* NVFP4: per-expert global scale = amax / denominator */
			tens a13 = LT("mlp.experts.w13_weight_amax"), d13 = LT("mlp.experts.w13_weight_outer_scale_denominator");
			tens a2 = LT("mlp.experts.w2_weight_amax"), d2 = LT("mlp.experts.w2_weight_outer_scale_denominator");
			for (int e = 0; e < NEXP; e++) {
				L->gu[e] = nvfp4(g13, s13, e, 2 * NFF, NE, 1, tf(a13, e) / tf(d13, 0));
				L->dn[e] = nvfp4(g2, s2, e, NE, NFF, 0, tf(a2, e) / tf(d2, 0));
			}
		} else for (int e = 0; e < NEXP; e++) { L->gu[e] = mxfp8(g13, s13, e, 2 * NFF, NE, 1); L->dn[e] = mxfp8(g2, s2, e, NE, NFF, 0); }
		if (swa) { L->anorm = small(LT("attn_norm.weight"), NE); L->kconv = conv(LT("attn.k_sconv.weight"), KVD, 0); L->vconv = conv(LT("attn.v_sconv.weight"), KVD, 0); }
		else { L->qn = small(LT("q_input_norm.weight"), NE); L->kn = small(LT("k_input_norm.weight"), NE); L->vn = small(LT("v_input_norm.weight"), NE);
			L->kconv = conv(LT("attn.k_sconv.static_w"), KVD, 1); L->vconv = conv(LT("attn.v_sconv.static_w"), KVD, 1); }
		L->fnorm = small(LT("mlp_norm.weight"), NE); L->qclip = small(LT("attn.q_l2_clip.scale"), NH * HD); L->rclip = small(LT("attn.rel_l2_clip.scale"), NH * NREL);
		{ tens r = LT("attn.rel_logits_proj.proj"); float *v = xmalloc(sizeof(float) * REL * NREL);	/* [16][1024] -> [1024][16] */
		  for (int a = 0; a < REL; a++) for (int b = 0; b < NREL; b++) v[a * NREL + b] = tf(r, (size_t)b * REL + a);
		  tens c = { (const uint8_t *)v, "F32" }; L->relb = small(c, REL * NREL); free(v); }
		L->apc = conv(LT("attn_sconv.weight"), NE, 0); L->fpc = conv(LT("mlp_sconv.weight"), NE, 0);
		if (i > 0) { tens q = T("model.llm.attn_res.attn_queries.%d", i - 1); L->narq = q.nd == 2 ? q.sh[0] : 1; L->arq = small(q, L->narq * NE); }
		L->frq = small(T("model.llm.attn_res.mlp_queries.%d", i), NE); L->ebias = small(LT("mlp.gate.bias"), NEXP); L->router = small(LT("mlp.gate.weight"), NEXP * NE);
		fprintf(stderr, "\rlayer %d/%d (%.0f s)", i + 1, NPHYS, now() - t0);
	}
	w.out = quant(T("model.llm.unembed.weight"), NVOCAB4, NE);
	static const char *dn[9][2] = { {"q", "self_attn.q_proj"}, {"k", "self_attn.k_proj"}, {"v", "self_attn.v_proj"}, {"o", "self_attn.o_proj"},
		{"kc", "self_attn.k_proj_ctx"}, {"vc", "self_attn.v_proj_ctx"}, {"up", "mlp.up_proj"}, {"gate", "mlp.gate_proj"}, {"down", "mlp.down_proj"} };
	for (int l = 0; l < DL; l++) {
		dlayer *D = &w.D[l]; mat *m[9] = { &D->q, &D->k, &D->v, &D->o, &D->kc, &D->vc, &D->up, &D->gate, &D->down };
		for (int k = 0; k < 9; k++) { tens t = T("layers.%d.%s.weight", l, dn[k][1]); *m[k] = quant(t, t.sh[0], t.sh[1]); }
		D->an = small(T("layers.%d.input_layernorm.weight", l), NE); D->qn = small(T("layers.%d.self_attn.q_norm.weight", l), HD);
		D->kn = small(T("layers.%d.self_attn.k_norm.weight", l), HD); D->kcn = small(T("layers.%d.self_attn.k_norm_ctx.weight", l), HD);
		D->fn = small(T("layers.%d.post_attention_layernorm.weight", l), NE); D->dfl = small(T("layers.%d.dflare_norm.weight", l), NE);
		D->swa = l != 1;
	}
	w.enorm = small(T("model.llm.embed_norm.weight"), NE); w.onorm = small(T("model.llm.norm.weight"), NE); w.oq = small(T("model.llm.attn_res.output_query"), NE);
	w.loopq = small(T("model.llm.middle_cycle_cross_pass_queries"), 24 * NE); w.loopkvq = small(T("model.llm.middle_cycle_cross_pass_kv_queries"), 24 * NE);
	w.fuse = small(T("fuse_weight"), 16); w.dnorm = small(T("markov_head.dflare_norm.weight"), NE); w.doutn = small(T("norm.weight"), NE);
	w.mask = small(T("mask_embed"), NE); w.jointb = small(T("markov_head.rnn_proj.bias"), 3 * DRANK); w.houtb = small(T("markov_head.hidden_out.bias"), NE);
	/* tail: smalls, q8 embedding tables, markov head, vocabulary */
	w.arena = A0; w.narena = apos; fpos = A0 + apos;
	w.smalls = blob(sm, nsm * 4); w.nsmalls = nsm;
	w.emb = q8tab(T("model.llm.embed.weight"), NVOCAB); w.unemb = q8tab(T("model.llm.unembed.weight"), NVOCAB);
	w.w1 = blob(T("markov_head.markov_w1.weight").p, (size_t)NVOCAB * DRANK * 2);
	{ tens j = T("markov_head.rnn_proj.weight"); size_t n = (size_t)3 * DRANK * (2 * DRANK + 2 * NE); __fp16 *h = xmalloc(n * 2);
	  for (size_t i = 0; i < n; i++) h[i] = (__fp16)tf(j, i);
	  w.joint = blob(h, n * 2); free(h); }
	{ tens h = T("markov_head.hidden_out.weight"); float *f = xmalloc(sizeof(float) * NE * DRANK);
	  for (size_t i = 0; i < (size_t)NE * DRANK; i++) f[i] = tf(h, i);
	  w.hout = blob(f, sizeof(float) * NE * DRANK); free(f); }
	{ snprintf(p, sizeof p, "%s/tiktoken/tokenizer.model", dir); FILE *f = fopen(p, "r"); if (!f) die("cannot open %s", p);
	  str bytes = { 0 }; uint32_t *off = xmalloc(4 * (NTOK + 1)); char line[4096]; int n = 0;
	  while (n < NTOK && fgets(line, sizeof line, f)) { off[n++] = (uint32_t)bytes.n; char *sp = strchr(line, ' '); if (sp) *sp = 0; b64(line, &bytes); }
	  fclose(f); if (n != NTOK) die("tokenizer: %d tokens", n);
	  off[NTOK] = (uint32_t)bytes.n;
	  w.vocab = blob(off, 4 * (NTOK + 1)); w.nvocab = bytes.n; blob(bytes.p, bytes.n); sfree(&bytes); free(off); }
	pw(&w, sizeof w, 0);
	if (ftruncate(ofd, fpos)) die("truncate failed");
	close(ofd);
	fprintf(stderr, "\nwrote %s: %.2f GB in %.0f s\n", out, fpos / 1e9, now() - t0);
	return 0;
}

/* ---------------- load ---------------- */
static void load(const char *path) {
	int fd = open(path, O_RDONLY); struct stat st;
	if (fd < 0 || fstat(fd, &st)) die("cannot open %s (run: air prep <checkpoint dir> %s)", path, path);
	static wfile w;
	if (pread(fd, &w, sizeof w, 0) != sizeof w || memcmp(w.magic, AIRMAGIC, 8)) die("%s: not an air weight file", path);
	W = &w;
	double t0 = now();
	/* the 4-bit weights are read (not mapped) into page-aligned memory: Metal wraps them without a copy */
	size_t na = (w.narena + 16383) & ~(size_t)16383;
	AR = mmap(NULL, na, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
	if (AR == MAP_FAILED) die("out of memory");
	fcntl(fd, F_NOCACHE, 1);
	for (size_t d = 0; d < w.narena; ) { ssize_t k = pread(fd, AR + d, w.narena - d > (1 << 30) ? 1 << 30 : w.narena - d, w.arena + d); if (k <= 0) die("read failed"); d += k; }
	uint8_t *m = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	if (m == MAP_FAILED) die("cannot map %s", path);
	SM = xmalloc(w.nsmalls * 4); memcpy(SM, m + w.smalls, w.nsmalls * 4);
	Q8E = m + w.emb; Q8U = m + w.unemb; W1 = (uint16_t *)(m + w.w1); JNT = (__fp16 *)(m + w.joint); HOUT = (float *)(m + w.hout);
	VOFF = (uint32_t *)(m + w.vocab); VOC = m + w.vocab + (((size_t)4 * (NTOK + 1) + 16383) & ~(size_t)16383);
	fprintf(stderr, "loaded %s (%.1f GB) in %.1f s\n", path, st.st_size / 1e9, now() - t0);
}

/* ================= tokenizer: o200k-style pre-split, byte-level BPE by rank ================= */
int8_t u_charType(int32_t c);
int8_t u_isUWhiteSpace(int32_t c);
enum { NBPE = 199998, HBITS = 20 };
static int *htab;
static uint32_t hsh(const uint8_t *s, int n) { uint32_t h = 2166136261u; for (int i = 0; i < n; i++) h = (h ^ s[i]) * 16777619u; return h; }
static int lookup(const uint8_t *s, int n) {
	for (uint32_t h = hsh(s, n) & ((1 << HBITS) - 1);; h = (h + 1) & ((1 << HBITS) - 1)) {
		int id = htab[h];
		if (id < 0) return -1;
		if ((int)(VOFF[id + 1] - VOFF[id]) == n && !memcmp(VOC + VOFF[id], s, n)) return id;
	}
}
static void tok_init(void) {
	htab = xmalloc(sizeof(int) << HBITS); memset(htab, 0xff, sizeof(int) << HBITS);
	for (int id = 0; id < NBPE; id++) {
		uint32_t h = hsh(VOC + VOFF[id], VOFF[id + 1] - VOFF[id]) & ((1 << HBITS) - 1);
		while (htab[h] >= 0) h = (h + 1) & ((1 << HBITS) - 1);
		htab[h] = id;
	}
}
static void bpe(const uint8_t *s, int n, ivec *out) {
	int id = lookup(s, n);
	if (id >= 0) { ipush(out, id); return; }
	int *b = xmalloc(sizeof(int) * (n + 1)), m = n;		/* piece boundaries */
	for (int i = 0; i <= n; i++) b[i] = i;
	while (m > 1) {
		int best = -1, bi = -1;
		for (int i = 0; i + 1 < m; i++) { int r = lookup(s + b[i], b[i + 2] - b[i]); if (r >= 0 && (best < 0 || r < best)) { best = r; bi = i; } }
		if (bi < 0) break;
		memmove(b + bi + 1, b + bi + 2, sizeof(int) * (m - bi - 1)); m--;
	}
	for (int i = 0; i < m; i++) ipush(out, lookup(s + b[i], b[i + 1] - b[i]));
	free(b);
}
static int isL(int c) { int t = u_charType(c); return t >= 1 && t <= 5; }
static int isN(int c) { int t = u_charType(c); return t >= 9 && t <= 11; }
static int isM(int c) { int t = u_charType(c); return t >= 6 && t <= 8; }
static int isU(int c) { int t = u_charType(c); return t == 1 || t == 3 || t == 4 || t == 5 || isM(c); }	/* Lu Lt Lm Lo M */
static int isW(int c) { int t = u_charType(c); return t == 2 || t == 4 || t == 5 || isM(c); }		/* Ll Lm Lo M */
static int isS(int c) { return u_isUWhiteSpace(c); }
static int contraction(const int *c, int n, int e) {
	if (e + 1 >= n || c[e] != '\'') return e;
	int a = tolower(c[e + 1] < 128 ? c[e + 1] : 0), b = e + 2 < n && c[e + 2] < 128 ? tolower(c[e + 2]) : 0;
	if (a == 's' || a == 't' || a == 'm' || a == 'd') return e + 2;
	if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return e + 3;
	return e;
}
/* end of the pre-token starting at i (the split regex, alternatives tried in order) */
static int split(const int *c, int n, int i) {
	int pfx = c[i] != '\r' && c[i] != '\n' && !isL(c[i]) && !isN(c[i]);
	for (int alt = 0; alt < 2; alt++) for (int k = pfx; k >= 0; k--) {
		int j = i + k, a = j; if (j >= n) continue;
		while (a < n && isU(c[a])) a++;
		if (alt == 0) {		/* [Lu Lt Lm Lo M]* [Ll Lm Lo M]+ */
			for (int s = a; s >= j; s--) if (s < n && isW(c[s])) { int e = s; while (e < n && isW(c[e])) e++; return contraction(c, n, e); }
		} else if (a > j) {	/* [Lu Lt Lm Lo M]+ [Ll Lm Lo M]* */
			int e = a; while (e < n && isW(c[e])) e++; return contraction(c, n, e);
		}
	}
	if (isN(c[i])) { int e = i + 1; while (e < n && e < i + 3 && isN(c[e])) e++; return e; }
	{ int j = i + (c[i] == ' '), e = j;
	  while (e < n && !isS(c[e]) && !isL(c[e]) && !isN(c[e])) e++;
	  if (e > j) { while (e < n && (c[e] == '\r' || c[e] == '\n' || c[e] == '/')) e++; return e; } }
	int w = i; while (w < n && isS(c[w])) w++;
	for (int k = w - 1; k >= i; k--) if (c[k] == '\r' || c[k] == '\n') return k + 1;
	if (w == n || w - i < 2) return w;
	return w - 1;
}
static void tokenize(const char *s, size_t len, ivec *out) {
	int *cp = xmalloc(sizeof(int) * (len + 1)), *at = xmalloc(sizeof(int) * (len + 1)), n = 0;
	for (size_t i = 0; i < len; ) {		/* utf-8 decode (invalid bytes pass through as themselves) */
		unsigned char c = s[i]; int k = c < 0x80 ? 1 : c >> 5 == 6 ? 2 : c >> 4 == 14 ? 3 : c >> 3 == 30 ? 4 : 1, v = k == 1 ? c : c & (0x3F >> (k - 1));
		if (i + k > len) k = 1, v = c;
		for (int j = 1; j < k; j++) v = v << 6 | (s[i + j] & 63);
		at[n] = (int)i; cp[n++] = v; i += k;
	}
	at[n] = (int)len;
	for (int i = 0; i < n; ) { int e = split(cp, n, i); if (e <= i) e = i + 1; bpe((const uint8_t *)s + at[i], at[e] - at[i], out); i = e; }
	free(cp); free(at);
}
static void tok(ivec *out, const char *s) { tokenize(s, strlen(s), out); }
static int piece(int id, const char **p) { if (id < 0 || id >= NBPE) return 0; *p = (const char *)VOC + VOFF[id]; return VOFF[id + 1] - VOFF[id]; }

/* ================= engine: one sequence, greedy speculative decoding with the DSpark drafter ================= */
static struct {
	int max_ctx, n, last, have;	/* committed positions, pending token */
	int *hist;
	struct { int pos, hi; float *conv, *ctx; } snap[8];
} E;
enum { NDRAFT = 3, SLACK = RING - SWA - MAXB };

/* forward B tokens at E.n; best[b] = argmax by exact (q8) rescoring of the 16 gpu candidates */
static void fwd(const int *t, int B, int last, int *best, float *bestv) {
	int cand[MAXB * TOPC]; float row[NE];
	gpu_forward(t, B, E.n, last, cand);
	const float *x = gpu_x();
	for (int b = last ? B - 1 : 0; b < B; b++) {
		float bv = -INFINITY; int bi = cand[b * TOPC];
		for (int k = 0; k < TOPC; k++) {
			int c = cand[b * TOPC + k]; if (c < 0 || c >= NVOCAB) continue;
			q8row(W->unemb, c, row); float v = dot(row, x + b * NE, NE);
			if (v > bv || (v == bv && c < bi)) { bv = v; bi = c; }
		}
		best[b] = bi; if (bestv) bestv[b] = bv;
	}
	for (int b = 0; b < B; b++) if (E.n + b < E.max_ctx + 64) E.hist[E.n + b] = t[b];
	E.n += B;
}
static void prefill(const int *t, int n) {
	int *all = xmalloc(sizeof(int) * (n + 1)), na = 0;
	if (E.have) all[na++] = E.last;
	memcpy(all + na, t, sizeof(int) * n); na += n; E.have = 0;
	if (na > 12) {		/* GEMM prefill in chunks of PMAX, then exact rescoring of the last row */
		int cand[TOPC], i = 0; float row[NE];
		while (i < na) {
			int b = na - i < PMAX ? na - i : PMAX;
			for (int k = 0; k < b; k++) E.hist[E.n + k] = all[i + k];
			gpu_prefill(all + i, b, E.n, cand); E.n += b; i += b;
			if (i == na) {
				const float *x = gpu_x() + (size_t)(b - 1) * NE; float bv = -INFINITY; int bi = cand[0];
				for (int k = 0; k < TOPC; k++) { int c = cand[k]; if (c < 0 || c >= NVOCAB) continue;
					q8row(W->unemb, c, row); float v = dot(row, x, NE); if (v > bv || (v == bv && c < bi)) { bv = v; bi = c; } }
				E.last = bi;
			}
		}
	} else for (int i = 0; i < na; ) {	/* short: decode path, 12 at a time (conv ring holds 16) */
		int b = na - i < 12 ? na - i : 12, best[MAXB], p0 = E.n;
		fwd(all + i, b, 1, best, NULL);
		gpu_draft(p0, b, 0, 0, 0, NULL, NULL);
		i += b; E.last = best[b - 1];
	}
	E.have = na > 0; free(all);
}

/* GRU markov head over the drafted rows: correction of each row's hidden, exact rescoring of its 64 candidates */
static void markov(int anchor, int pos0, int nd, const float *xr, const int *cand, int *out, float *margin) {
	float h[DRANK] = { 0 }, in[2 * DRANK + 2 * NE], jo[3 * DRANK], lat[DRANK], hid[NE], row[NE];
	const float *ctx = gpu_ctxring() + (size_t)((pos0 - 1) % CTXRING) * NE;
	const int K = 2 * DRANK + 2 * NE;
	int prev = anchor;
	for (int i = 0; i < nd; i++) {
		const float *x = xr + (size_t)i * NE;
		memcpy(in, h, sizeof h);
		for (int c = 0; c < DRANK; c++) in[DRANK + c] = bf16(W1[(size_t)prev * DRANK + c]);
		memcpy(in + 2 * DRANK, x, NE * 4); memcpy(in + 2 * DRANK + NE, ctx, NE * 4);
		for (int r = 0; r < 3 * DRANK; r++) {
			const __fp16 *w = JNT + (size_t)r * K; float32x4_t a0 = vdupq_n_f32(0), a1 = a0;
			for (int k = 0; k < K; k += 8) { float16x8_t v = vld1q_f16(w + k);
				a0 = vfmaq_f32(a0, vcvt_f32_f16(vget_low_f16(v)), vld1q_f32(in + k)); a1 = vfmaq_f32(a1, vcvt_f32_f16(vget_high_f16(v)), vld1q_f32(in + k + 4)); }
			jo[r] = vaddvq_f32(vaddq_f32(a0, a1)) + SM[W->jointb + r];
		}
		for (int c = 0; c < DRANK; c++) { float g = sigm(jo[c]), cn = tanhf(jo[DRANK + c]); h[c] = cn + g * (h[c] - cn); lat[c] = tanhf(jo[2 * DRANK + c]); }
		for (int c = 0; c < NE; c++) hid[c] = x[c] + dot(HOUT + (size_t)c * DRANK, lat, DRANK) + SM[W->houtb + c];
		float bv = -INFINITY, sv = -INFINITY; int best = cand[i * DCAND];
		for (int k = 0; k < DCAND; k++) {
			int c = cand[i * DCAND + k]; if (c < 0 || c >= NVOCAB) continue;
			q8row(W->unemb, c, row); float v = dot(row, hid, NE);
			if (v > bv || (v == bv && c < best)) { sv = bv; bv = v; best = c; } else if (v > sv) sv = v;
		}
		out[i] = prev = best; margin[i] = bv - sv;
		if (bv - sv < 2 && i + 1 < nd) { for (int j = i + 1; j < nd; j++) { out[j] = best; margin[j] = 0; } break; }	/* unsure: stop drafting */
	}
}
static void draft(int ipos, int ni, int anchor, int pos0, int nd, int *out, float *margin) {
	int cand[DBLOCK * DCAND]; const float *x;
	gpu_draft(ipos, ni, anchor, pos0, nd, cand, &x);
	if (nd > 0) markov(anchor, pos0, nd, x, cand, out, margin);
}

/* adaptive draft length: acceptance of draft slot i given its drafter margin (calibrated), verify cost per batch size */
static const float MB_[8] = { 0, 0.5f, 1, 2, 3, 5, 8, 1e9f };
static const float PACC[3][7] = { { 0.19f, 0.26f, 0.33f, 0.39f, 0.56f, 0.72f, 0.97f },
				  { 0.14f, 0.27f, 0.27f, 0.39f, 0.49f, 0.69f, 0.94f },
				  { 0.10f, 0.05f, 0.14f, 0.08f, 0.31f, 0.76f, 0.93f } };
static double cost[MAXB + 1] = { 0, 45, 58, 72, 88, 100, 110, 120, 130 };	/* ms per verify of B rows (running average) */
static int choose(const float *mg, int nmax) {
	double bv = -1, et = 1, pr = 1; int bl = 0;
	for (int L = 0; L <= nmax; L++) {
		if (L > 0) { int s = L - 1 > 2 ? 2 : L - 1, b = 0; while (b < 6 && mg[L - 1] >= MB_[b + 1]) b++; pr *= PACC[s][b]; et += pr; }
		double v = et / (cost[L + 1] + 8.0);
		if (v > bv) { bv = v; bl = L; }
	}
	return bl;
}
/* emit tokens from the pending one until EOS, max_new, the context end, or cb returning nonzero */
static int generate(int max_new, int (*cb)(void *, int), void *ctx) {
	if (!E.have) return 0;
	int n = 1, pend[MAXB], havep = 0; float mg[8] = { 0 }, pmg[8] = { 0 };
	if (cb(ctx, E.last) || E.last == T_EOS) return n;
	while (n < max_new && E.n + NDRAFT + 1 < E.max_ctx) {
		int vb[MAXB], best[MAXB], pos0 = E.n;
		vb[0] = E.last;
		if (havep) { memcpy(vb + 1, pend, sizeof(int) * NDRAFT); memcpy(mg, pmg, sizeof mg); }
		else draft(0, 0, E.last, pos0, NDRAFT, vb + 1, mg);
		int L = choose(mg, NDRAFT);
		double t0 = now();
		fwd(vb, L + 1, 0, best, NULL);
		cost[L + 1] = 0.9 * cost[L + 1] + 0.1 * (now() - t0) * 1e3;
		int acc = 0, stop = 0;
		while (acc < L && vb[acc + 1] == best[acc]) acc++;
		for (int i = 1; i <= acc && !stop; i++) { n++; if (cb(ctx, vb[i]) || vb[i] == T_EOS) stop = 1; }
		E.n = pos0 + 1 + acc;			/* roll back the rejected rows */
		if (stop) { E.have = 0; draft(pos0, L + 1, 0, 0, 0, NULL, NULL); break; }
		E.last = best[acc]; E.have = 1; n++;
		int fin = cb(ctx, E.last) || E.last == T_EOS || n >= max_new;
		draft(pos0, L + 1, E.last, E.n, fin ? 0 : NDRAFT, pend, pmg);	/* inject this step, draft the next */
		havep = !fin;
		if (fin) break;
	}
	return n;
}

/* ---- prefix cache: roll the state back to an earlier position ---- */
static void snapshot(void) {
	int pos = E.n, k = -1;
	for (int i = 0; i < 8; i++) if (E.snap[i].conv && E.snap[i].pos == pos) k = i;
	for (int i = 0; k < 0 && i < 8; i++) if (!E.snap[i].conv) k = i;
	if (k < 0) { k = 0; for (int i = 1; i < 8; i++) {	/* evict: stale first, then the oldest position */
		int si = E.snap[i].hi - E.snap[i].pos > SLACK, sk = E.snap[k].hi - E.snap[k].pos > SLACK;
		if (si > sk || (si == sk && E.snap[i].pos < E.snap[k].pos)) k = i; } }
	if (!E.snap[k].conv) { E.snap[k].conv = xmalloc(sizeof(float) * NVIS * 3 * CONVW); E.snap[k].ctx = xmalloc(sizeof(float) * CTXRING * NE); }
	E.snap[k].pos = pos; E.snap[k].hi = pos + MAXB;
	for (int v = 0; v < NVIS; v++) for (int j = 0; j < 3; j++) if (pos - 3 + j >= 0)
		memcpy(E.snap[k].conv + ((size_t)v * 3 + j) * CONVW, gpu_conv(v) + (size_t)((pos - 3 + j) % HIST) * CONVW, sizeof(float) * CONVW);
	memcpy(E.snap[k].ctx, gpu_ctxring(), sizeof(float) * CTXRING * NE);
}
static void dropsnaps(int above) {
	for (int i = 0; i < 8; i++) if (E.snap[i].conv && E.snap[i].pos > above) { free(E.snap[i].conv); free(E.snap[i].ctx); E.snap[i].conv = E.snap[i].ctx = NULL; }
}
/* keep the longest prefix of t[0..n) the state can roll back to (directly, by snapshot, or from scratch); returns its length (< n) */
static int keep(const int *t, int n, int dry) {
	int c = E.n, l = 0, k = -1;
	while (l < c && l < n && E.hist[l] == t[l]) l++;
	int r = l < n - 1 ? l : n - 1;
	if (r < 0) r = 0;
	if (c - r > 12) {		/* the conv ring (HIST) covers short rollbacks only */
		for (int i = 0; i < 8; i++) { if (!E.snap[i].conv || E.snap[i].pos > r) continue;
			int hi = E.snap[i].hi > c + MAXB ? E.snap[i].hi : c + MAXB;
			if (hi - E.snap[i].pos <= SLACK && (k < 0 || E.snap[i].pos > E.snap[k].pos)) k = i; }
		r = k >= 0 ? E.snap[k].pos : 0;
	}
	if (dry) return r;
	E.have = 0;
	for (int i = 0; i < 8; i++) if (E.snap[i].conv && c + MAXB > E.snap[i].hi) E.snap[i].hi = c + MAXB;
	if (k >= 0) {
		for (int v = 0; v < NVIS; v++) for (int j = 0; j < 3; j++) if (r - 3 + j >= 0)
			memcpy(gpu_conv(v) + (size_t)((r - 3 + j) % HIST) * CONVW, E.snap[k].conv + ((size_t)v * 3 + j) * CONVW, sizeof(float) * CONVW);
		memcpy(gpu_ctxring(), E.snap[k].ctx, sizeof(float) * CTXRING * NE);
	}
	E.n = r; dropsnaps(r);
	return r;
}

/* ---- on-disk prefix checkpoints: full state at shared-prefix boundaries (tools, system prompt) ---- */
enum { NCK = 6, CKMIN = 4096 };
static struct { char path[600]; int *t, n; double used; } CK[NCK];
static int nck;
static char ckdir[512] = "cache";
static void ck_index(void) {
	mkdir(ckdir, 0755);
	DIR *d = opendir(ckdir); struct dirent *de;
	while (d && (de = readdir(d)) && nck < NCK) {
		char p[700]; snprintf(p, sizeof p, "%s/%s", ckdir, de->d_name);
		FILE *f = strstr(de->d_name, ".ck") ? fopen(p, "rb") : NULL; int h[2];
		if (!f) continue;
		if (fread(h, 4, 2, f) == 2 && h[0] == 0x31726961 && h[1] > 0 && h[1] < E.max_ctx) {
			CK[nck].t = xmalloc(sizeof(int) * h[1]);
			if (fread(CK[nck].t, 4, h[1], f) == (size_t)h[1]) { CK[nck].n = h[1]; snprintf(CK[nck].path, sizeof CK[0].path, "%s", p); nck++; }
			else free(CK[nck].t);
		}
		fclose(f);
	}
	if (d) closedir(d);
}
static int ck_best(const int *t, int n) {
	int b = -1;
	for (int i = 0; i < nck; i++) if (CK[i].n < n && !memcmp(CK[i].t, t, sizeof(int) * CK[i].n) && (b < 0 || CK[i].n > CK[b].n)) b = i;
	return b;
}
static int ck_has(const int *t, int n) { for (int i = 0; i < nck; i++) if (CK[i].n == n && !memcmp(CK[i].t, t, sizeof(int) * n)) return 1; return 0; }
static void ck_save(void) {
	double t0 = now(); int k = nck < NCK ? nck++ : 0;
	for (int i = 1; nck == NCK && i < NCK; i++) if (CK[i].used < CK[k].used) k = i;
	if (CK[k].t) { unlink(CK[k].path); free(CK[k].t); }
	uint32_t h = 2166136261u; for (int i = 0; i < E.n; i++) h = (h ^ (uint32_t)E.hist[i]) * 16777619u;
	snprintf(CK[k].path, sizeof CK[k].path, "%s/%d-%08x.ck", ckdir, E.n, h);
	FILE *f = fopen(CK[k].path, "wb"); int hd[2] = { 0x31726961, E.n };
	int ok = f && fwrite(hd, 4, 2, f) == 2 && fwrite(E.hist, 4, E.n, f) == (size_t)E.n && gpu_state(f, E.n, 1) == 0;
	if (f) fclose(f);
	CK[k].t = xmalloc(sizeof(int) * E.n); memcpy(CK[k].t, E.hist, sizeof(int) * E.n); CK[k].n = E.n; CK[k].used = now();
	if (!ok) { unlink(CK[k].path); CK[k].n = 0; }
	fprintf(stderr, "[air] checkpoint %d tokens (%.2f s)\n", E.n, now() - t0);
}
static int ck_load(int k) {
	FILE *f = fopen(CK[k].path, "rb"); int hd[2];
	int ok = f && fread(hd, 4, 2, f) == 2 && hd[1] == CK[k].n && fseek(f, 4L * hd[1], SEEK_CUR) == 0 && gpu_state(f, CK[k].n, 0) == 0;
	if (f) fclose(f);
	dropsnaps(-1); E.have = 0;
	if (!ok) { E.n = 0; return -1; }
	E.n = CK[k].n; memcpy(E.hist, CK[k].t, sizeof(int) * E.n); CK[k].used = now();
	return 0;
}

/* ================= chat template ================= */
static float effort0 = 0.1f, think0 = 0.7f;
static void msg(ivec *o, int role, const char *name, int content, const char *s, size_t n) {
	ipush(o, role); if (name) tok(o, name); ipush(o, content); tokenize(s, n, o); ipush(o, T_END);
}
static void effort_msg(ivec *o, float e) {
	char b[64]; snprintf(b, sizeof b, e == 0 ? "Thinking effort level: 0" : "Thinking effort level: %g", e);
	msg(o, T_SYS, NULL, T_TEXT, b, strlen(b));
}
/* assistant turns this process generated, by visible content -> their exact tokens (so a resent conversation matches the cache) */
enum { NSPL = 64 };
static struct { char *key; int *t, n; } SPL[NSPL];
static int nspl;
static void key_text(str *k, const char *s, size_t n) {
	while (n && isspace((unsigned char)*s)) s++, n--;
	while (n && isspace((unsigned char)s[n - 1])) n--;
	if (n) { sputs(k, "T\1"); sadd(k, s, n); sputs(k, "\2"); }
}
static void key_tool(str *k, const char *name, const J *in) { sputs(k, "U\1"); sputs(k, name); sputs(k, "\1"); if (in) jdump(k, in, 1); else sputs(k, "{}"); sputs(k, "\2"); }
static int spl_get(const char *key) { for (int i = 0; i < NSPL; i++) if (SPL[i].key && !strcmp(SPL[i].key, key)) return i; return -1; }
static void spl_put(const char *key, const int *t, int n) {
	if (spl_get(key) >= 0) return;
	int i = nspl++ % NSPL; free(SPL[i].key); free(SPL[i].t);
	SPL[i].key = strdup(key); SPL[i].t = xmalloc(sizeof(int) * (n + 1)); memcpy(SPL[i].t, t, sizeof(int) * n); SPL[i].n = n;
}
static void call_json(str *b, const char *name, const J *in) {	/* {"name":..,"args":{sorted keys}} */
	sputs(b, "{\"name\":"); sjson(b, name, strlen(name)); sputs(b, ",\"args\":{");
	if (in && in->t == JOBJ) { int *o = jorder(in, 1);
		for (int i = 0; i < in->n; i++) { if (i) sputs(b, ","); sjson(b, in->k[o[i]], in->kl[o[i]]); sputs(b, ":"); jdump(b, in->a[o[i]], 0); }
		free(o); }
	sputs(b, "}}");
}
static void text_of(str *o, const J *c) {	/* tool_result content: string or blocks */
	if (c && c->t == JSTR) { sadd(o, c->s, c->len); return; }
	for (int i = 0; c && c->t == JARR && i < c->n; i++) {
		const char *ty = jsv(c->a[i], "type"), *tx = jsv(c->a[i], "text");
		if (o->n) sputs(o, "\n");
		if (tx) sputs(o, tx); else if (ty && !strcmp(ty, "image")) sputs(o, "[image omitted: text-only server]");
	}
}
static const char *tool_name(const J *msgs, const char *id) {
	for (int m = 0; id && m < msgs->n; m++) { J *c = jget(msgs->a[m], "content");
		for (int i = 0; c && c->t == JARR && i < c->n; i++) { const char *ty = jsv(c->a[i], "type"), *bid = jsv(c->a[i], "id");
			if (ty && bid && !strcmp(ty, "tool_use") && !strcmp(bid, id)) { const char *n = jsv(c->a[i], "name"); return n ? n : ""; } } }
	return "";
}
static void sysmsg(ivec *o, const J *c) {
	if (c && c->t == JSTR && c->len) msg(o, T_SYS, NULL, T_TEXT, c->s, c->len);
	for (int i = 0; c && c->t == JARR && i < c->n; i++) { const char *t = jsv(c->a[i], "text");
		if (t && strncmp(t, "x-anthropic-billing-header:", 27)) msg(o, T_SYS, NULL, T_TEXT, t, strlen(t)); }	/* skip per-request client metadata */
}
/* Anthropic request -> token ids. bnd: checkpoint boundaries (end of tools, of system, of the first message's shared blocks) */
static int render(const J *rq, ivec *o, int *bnd, char *err, size_t errn) {
	J *tools = jget(rq, "tools"), *msgs = jget(rq, "messages");
	if (!msgs || msgs->t != JARR) { snprintf(err, errn, "messages: expected an array"); return -1; }
	str b = { 0 }; int nt = 0;
	for (int i = 0; tools && tools->t == JARR && i < tools->n; i++) {
		J *t = tools->a[i], *sc = jget(t, "input_schema"); const char *nm = jsv(t, "name"), *d = jsv(t, "description");
		if (!nm || !sc) continue;		/* server tools have no schema */
		sputs(&b, nt++ ? "," : "["); sputs(&b, "{\"description\":"); sjson(&b, d ? d : "", d ? strlen(d) : 0);
		sputs(&b, ",\"name\":"); sjson(&b, nm, strlen(nm)); sputs(&b, ",\"parameters\":"); jdump(&b, sc, 1); sputs(&b, ",\"type\":\"function\"}");
	}
	if (nt) { sputs(&b, "]"); msg(o, T_SYS, "tool_declare", T_XML, b.p, b.n); }
	sfree(&b);
	bnd[0] = o->n;
	sysmsg(o, jget(rq, "system"));
	int fa = msgs->n, fu = -1;		/* system messages before the first assistant turn join the system prefix */
	for (int m = 0; m < msgs->n; m++) { const char *r = jsv(msgs->a[m], "role"); if (r && !strcmp(r, "assistant")) { fa = m; break; } }
	for (int m = 0; m < fa; m++) { const char *r = jsv(msgs->a[m], "role"); if (r && !strcmp(r, "system")) sysmsg(o, jget(msgs->a[m], "content")); }
	{ J *th = jget(rq, "thinking"); const char *tt = jsv(th, "type"); float e = tt && !strcmp(tt, "enabled") ? think0 : effort0;
	  J *re = jget(rq, "reasoning_effort");
	  if (re && re->t == JNUM) e = atof(re->s);
	  if (re && re->t == JSTR) { static const char *nm[] = { "none", "minimal", "low", "medium", "high", "xhigh", "max" }; static const float vl[] = { 0, 0.1f, 0.2f, 0.7f, 0.9f, 0.99f, 0.99f };
		for (int i = 0; i < 7; i++) if (!strcmp(re->s, nm[i])) e = vl[i]; }
	  effort_msg(o, e < 0 ? 0 : e > 0.99f ? 0.99f : e); }
	bnd[1] = o->n; bnd[2] = 0;
	for (int m = 0; m < msgs->n; m++) {
		J *mm = msgs->a[m], *c = jget(mm, "content"); const char *role = jsv(mm, "role");
		if (!role) { snprintf(err, errn, "messages.%d: missing role", m); return -1; }
		if (!strcmp(role, "system")) { if (m > fa) sysmsg(o, c); continue; }
		if (!strcmp(role, "user")) {
			if (fu < 0) fu = m;
			if (c && c->t == JSTR) { msg(o, T_USER, NULL, T_TEXT, c->s, c->len); continue; }
			for (int i = 0; c && c->t == JARR && i < c->n; i++) {
				J *bl = c->a[i]; const char *ty = jsv(bl, "type");
				if (!ty) continue;
				if (!strcmp(ty, "text")) { const char *t = jsv(bl, "text"); msg(o, T_USER, NULL, T_TEXT, t ? t : "", t ? strlen(t) : 0); if (m == fu && i == c->n - 2) bnd[2] = o->n; }
				else if (!strcmp(ty, "tool_result")) {
					str t = { 0 }; sadd(&t, "", 0); J *ie = jget(bl, "is_error");
					if (ie && ie->t == JTRUE) sputs(&t, "Error: ");
					text_of(&t, jget(bl, "content"));
					msg(o, T_TOOL, tool_name(msgs, jsv(bl, "tool_use_id")), T_TEXT, t.p, t.n); sfree(&t);
				} else if (!strcmp(ty, "image") || !strcmp(ty, "document")) { const char *t = "[attachment omitted: text-only server]"; msg(o, T_USER, NULL, T_TEXT, t, strlen(t)); }
			}
		} else if (!strcmp(role, "assistant")) {
			int lastm = m == msgs->n - 1; str k = { 0 }; sadd(&k, "", 0);
			if (c && c->t == JSTR) key_text(&k, c->s, c->len);
			for (int i = 0; c && c->t == JARR && i < c->n; i++) { J *bl = c->a[i]; const char *ty = jsv(bl, "type"); J *tx = jget(bl, "text");
				if (ty && !strcmp(ty, "text") && tx && tx->t == JSTR) key_text(&k, tx->s, tx->len);
				if (ty && !strcmp(ty, "tool_use")) { const char *nm = jsv(bl, "name"); key_tool(&k, nm ? nm : "", jget(bl, "input")); } }
			int si = lastm ? -1 : spl_get(k.p); sfree(&k);
			if (si >= 0) { for (int i = 0; i < SPL[si].n; i++) ipush(o, SPL[si].t[i]); if (!SPL[si].n || SPL[si].t[SPL[si].n - 1] != T_EOS) ipush(o, T_EOS); continue; }
			if (c && c->t == JSTR && c->len) msg(o, T_MODEL, NULL, T_TEXT, c->s, c->len);
			for (int i = 0; c && c->t == JARR && i < c->n; i++) {
				J *bl = c->a[i], *x; const char *ty = jsv(bl, "type");
				if (!ty) continue;
				if (!strcmp(ty, "thinking") && (x = jget(bl, "thinking")) && x->t == JSTR && x->len) msg(o, T_MODEL, NULL, T_THINK, x->s, x->len);
				else if (!strcmp(ty, "text") && (x = jget(bl, "text")) && x->t == JSTR && x->len) msg(o, T_MODEL, NULL, T_TEXT, x->s, x->len);
				else if (!strcmp(ty, "tool_use")) { const char *nm = jsv(bl, "name"); str a = { 0 }; call_json(&a, nm ? nm : "", jget(bl, "input"));
					msg(o, T_MODEL, nm ? nm : "", T_TOOLJSON, a.p, a.n); sfree(&a); }
			}
			if (!lastm) ipush(o, T_EOS);
		} else { snprintf(err, errn, "messages.%d: unsupported role %s", m, role); return -1; }
	}
	return 0;
}

/* ================= generated tokens -> Anthropic content blocks (streamed as SSE or collected) ================= */
typedef struct {
	int fd, stream, gone, nblk, open, state, tool, nout;	/* state: 0 between messages, 1 header, 2 text, 3 thinking, 4 tool json */
	str name, pend, json, content, key, cur, tname;
	ivec gen;
	char id[40], tid[40];
} gen;
static int wall(int fd, const char *p, size_t n) { while (n) { ssize_t k = write(fd, p, n); if (k <= 0) return -1; p += k; n -= k; } return 0; }
static void sse(gen *g, const char *ev, const char *data, size_t n) {
	if (g->gone || !g->stream) return;
	str b = { 0 }; sprint(&b, "event: %s\ndata: ", ev); sadd(&b, data, n); sputs(&b, "\n\n");
	if (wall(g->fd, b.p, b.n)) g->gone = 1;
	sfree(&b);
}
static void bstart(gen *g, int kind, const char *name) {
	str d = { 0 }, e = { 0 };
	g->open = 1; sfree(&g->cur); sadd(&g->cur, "", 0);
	if (kind == 2) sputs(&d, "{\"type\":\"text\",\"text\":\"\"}");
	else if (kind == 3) sputs(&d, "{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}");
	else { snprintf(g->tid, sizeof g->tid, "toolu_%08x%08x", arc4random(), arc4random()); sfree(&g->tname); sputs(&g->tname, name);
		sprint(&d, "{\"type\":\"tool_use\",\"id\":\"%s\",\"name\":", g->tid); sjson(&d, name, strlen(name)); sputs(&d, ",\"input\":{}}"); }
	sprint(&e, "{\"type\":\"content_block_start\",\"index\":%d,\"content_block\":", g->nblk); sadd(&e, d.p, d.n); sputs(&e, "}");
	sse(g, "content_block_start", e.p, e.n); sfree(&d); sfree(&e);
}
static void bdelta(gen *g, const char *s, size_t n, int think) {
	if (!n) return;
	sadd(&g->cur, s, n);
	str e = { 0 }; sprint(&e, "{\"type\":\"content_block_delta\",\"index\":%d,\"delta\":{\"type\":\"%s\",\"%s\":", g->nblk, think ? "thinking_delta" : "text_delta", think ? "thinking" : "text");
	sjson(&e, s, n); sputs(&e, "}}"); sse(g, "content_block_delta", e.p, e.n); sfree(&e);
}
static void bstop(gen *g, int kind, const J *in) {
	str e = { 0 };
	if (g->content.n > 1) sputs(&g->content, ",");
	if (kind == 4) {
		str a = { 0 }; jdump(&a, in, 0);
		sprint(&e, "{\"type\":\"content_block_delta\",\"index\":%d,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":", g->nblk);
		sjson(&e, a.p, a.n); sputs(&e, "}}"); sse(g, "content_block_delta", e.p, e.n); e.n = 0;
		sprint(&g->content, "{\"type\":\"tool_use\",\"id\":\"%s\",\"name\":", g->tid); sjson(&g->content, g->tname.p, g->tname.n);
		sputs(&g->content, ",\"input\":"); sadd(&g->content, a.p, a.n); sputs(&g->content, "}"); sfree(&a);
	} else {
		if (kind == 3) { sprint(&e, "{\"type\":\"content_block_delta\",\"index\":%d,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"air\"}}", g->nblk);
			sse(g, "content_block_delta", e.p, e.n); e.n = 0; }
		sputs(&g->content, kind == 3 ? "{\"type\":\"thinking\",\"thinking\":" : "{\"type\":\"text\",\"text\":");
		sjson(&g->content, g->cur.p, g->cur.n); sputs(&g->content, kind == 3 ? ",\"signature\":\"air\"}" : "}");
		if (kind == 2) key_text(&g->key, g->cur.p, g->cur.n);
	}
	sprint(&e, "{\"type\":\"content_block_stop\",\"index\":%d}", g->nblk); sse(g, "content_block_stop", e.p, e.n); sfree(&e);
	g->nblk++; g->open = 0;
}
static size_t utf8ok(const char *p, size_t n) {		/* longest prefix ending on a character boundary */
	size_t i = n; int back = 0;
	while (i > 0 && back < 4 && ((unsigned char)p[i - 1] & 0xC0) == 0x80) i--, back++;
	if (!i) return n;
	unsigned char c = p[i - 1]; int need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
	return need == back + 1 ? n : i - 1;
}
static void flush(gen *g, int all) {
	size_t k = all ? g->pend.n : utf8ok(g->pend.p, g->pend.n);
	if (!k) return;
	bdelta(g, g->pend.p, k, g->state == 3);
	memmove(g->pend.p, g->pend.p + k, g->pend.n - k); g->pend.n -= k;
}
static void endblock(gen *g) {
	if (g->state == 2 || g->state == 3) { flush(g, 1); bstop(g, g->state, NULL); }
	else if (g->state == 4) {
		J *o = jparse(g->json.p ? g->json.p : "", g->json.n), *a = jget(o, "args");
		const char *nm = jsv(o, "name"); if (!nm || !*nm) nm = g->name.p ? g->name.p : "";
		if (!a) a = jget(o, "arguments");
		if (a && a->t == JOBJ) { bstart(g, 4, nm); bstop(g, 4, a); key_tool(&g->key, nm, a); g->tool = 1; }
		else { bstart(g, 2, NULL); bdelta(g, "[unparsed tool call] ", 21, 0); bdelta(g, g->json.p ? g->json.p : "", g->json.n, 0); bstop(g, 2, NULL); }
		jfree(o);
	}
	g->state = 0;
}
static int ontok(void *c, int t) {
	gen *g = c; const char *p; int n;
	ipush(&g->gen, t);
	if (t == T_MODEL) { endblock(g); g->state = 1; sfree(&g->name); sadd(&g->name, "", 0); }
	else if (t == T_TEXT || t == T_THINK) { if (g->state != 1) endblock(g); g->state = t == T_TEXT ? 2 : 3; g->pend.n = 0; bstart(g, g->state, NULL); }
	else if (t == T_TOOLJSON || t == T_TOOLTEXT) { if (g->state != 1) endblock(g); g->state = 4; sfree(&g->json); sadd(&g->json, "", 0); }
	else if (t == T_END || t == T_EOS) endblock(g);
	else if ((n = piece(t, &p)) > 0) {
		if (g->state == 0) { g->state = 2; g->pend.n = 0; bstart(g, 2, NULL); }
		if (g->state == 1) sadd(&g->name, p, n);
		else if (g->state == 4) sadd(&g->json, p, n);
		else { sadd(&g->pend, p, n); flush(g, 0); }
		g->nout++;
	}
	if (g->stream && (g->gen.n & 7) == 0) { char x; if (recv(g->fd, &x, 1, MSG_PEEK | MSG_DONTWAIT) == 0) g->gone = 1; }
	return g->gone;
}

/* ================= http server ================= */
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static const char *mname = "inkling-2-mini";
static int stubhaiku = 1;
static int respond(int fd, int code, const char *type, const char *body, size_t n) {
	str h = { 0 };
	sprint(&h, "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: keep-alive\r\n\r\n", code, code == 200 ? "OK" : code == 404 ? "Not Found" : "Bad Request", type, n);
	int r = wall(fd, h.p, h.n) || wall(fd, body, n); sfree(&h); return r;
}
static void error(int fd, int code, const char *type, const char *m) {
	str b = { 0 }; sputs(&b, "{\"type\":\"error\",\"error\":{\"type\":"); sjson(&b, type, strlen(type)); sputs(&b, ",\"message\":"); sjson(&b, m, strlen(m)); sputs(&b, "}}");
	respond(fd, code, "application/json", b.p, b.n); sfree(&b);
}
static const char *SSEHDR = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
static int stub(int fd, const char *model, int stream) {	/* side calls (haiku): answered without the model */
	str b = { 0 };
	if (!stream) {
		sputs(&b, "{\"id\":\"msg_stub\",\"type\":\"message\",\"role\":\"assistant\",\"model\":"); sjson(&b, model, strlen(model));
		sputs(&b, ",\"content\":[{\"type\":\"text\",\"text\":\"OK\"}],\"stop_reason\":\"end_turn\",\"stop_sequence\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}");
		respond(fd, 200, "application/json", b.p, b.n); sfree(&b); return 0;
	}
	sputs(&b, SSEHDR); sputs(&b, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_stub\",\"type\":\"message\",\"role\":\"assistant\",\"model\":");
	sjson(&b, model, strlen(model));
	sputs(&b, ",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n"
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n"
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"OK\"}}\n\n"
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n"
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":1}}\n\n"
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");
	wall(fd, b.p, b.n); sfree(&b); return 1;
}
/* returns 1 if the connection must close */
static int messages(int fd, const J *rq) {
	J *st = jget(rq, "stream"); int stream = st && st->t == JTRUE;
	const char *model = jsv(rq, "model"); if (!model) model = mname;
	if (stubhaiku && strcasestr(model, "haiku")) return stub(fd, model, stream);
	ivec t = { 0 }; int bnd[3]; char err[256];
	pthread_mutex_lock(&mu);
	if (render(rq, &t, bnd, err, sizeof err) || t.n + 64 > E.max_ctx) {
		pthread_mutex_unlock(&mu);
		if (t.n + 64 > E.max_ctx) snprintf(err, sizeof err, "prompt is too long: %d tokens > %d maximum", t.n, E.max_ctx - 64);
		error(fd, 400, "invalid_request_error", err); free(t.p); return 0;
	}
	int n = t.n, maxn = (int)jnum(rq, "max_tokens", 4096);
	if (maxn > E.max_ctx - n - 32) maxn = E.max_ctx - n - 32;
	if (maxn < 1) maxn = 1;
	double t0 = now();
	int k = ck_best(t.p, n);
	if (k >= 0 && CK[k].n > keep(t.p, n, 1) + 512) { if (ck_load(k)) fprintf(stderr, "[air] checkpoint load failed\n"); else fprintf(stderr, "[air] loaded checkpoint %d tokens (%.2f s)\n", CK[k].n, now() - t0); }
	int reused = keep(t.p, n, 0);
	gen g = { .fd = fd, .stream = stream }; sadd(&g.content, "[", 1); sadd(&g.key, "", 0);
	snprintf(g.id, sizeof g.id, "msg_%08x%08x", arc4random(), arc4random());
	if (stream) {
		str e = { 0 };
		if (wall(fd, SSEHDR, strlen(SSEHDR))) g.gone = 1;
		sprint(&e, "{\"type\":\"message_start\",\"message\":{\"id\":\"%s\",\"type\":\"message\",\"role\":\"assistant\",\"model\":", g.id); sjson(&e, model, strlen(model));
		sprint(&e, ",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":%d,\"cache_read_input_tokens\":%d,\"cache_creation_input_tokens\":0,\"output_tokens\":0}}}", n - reused, reused);
		sse(&g, "message_start", e.p, e.n); sfree(&e);
	}
	fprintf(stderr, "[air] prompt %d tokens, %d cached\n", n, reused);
	for (int i = reused; i < n && !g.gone; ) {	/* chunks: keepalive pings, checkpoints at shared-prefix boundaries, abort if the client left */
		int b = n - i < PMAX ? n - i : PMAX, cut = 0;
		for (int q = 0; q < 3; q++) if (bnd[q] > i && bnd[q] <= i + b && bnd[q] >= CKMIN && bnd[q] < n - 1 && !ck_has(t.p, bnd[q])) { cut = bnd[q]; break; }
		if (cut) b = cut - i;
		if (b > 1 && i + b == n) b--;		/* stop one short: snapshot at n-1 so a resent prompt hits */
		prefill(t.p + i, b); i += b;
		if (i < n) E.have = 0;		/* only the last chunk's prediction is the pending token */
		if (cut && i == cut) ck_save();
		if (i == n - 1) { snapshot(); prefill(t.p + i, 1); i++; }
		if (stream) sse(&g, "ping", "{\"type\":\"ping\"}", 15); else { char x; if (recv(fd, &x, 1, MSG_PEEK | MSG_DONTWAIT) == 0) g.gone = 1; }
	}
	double t1 = now(); int out = 0;
	if (!g.gone && E.have) out = generate(maxn, ontok, &g);
	if (g.state) endblock(&g);
	snapshot();
	double t2 = now();
	if (!g.gone) spl_put(g.key.p, g.gen.p, g.gen.n);
	int eos = g.gen.n && g.gen.p[g.gen.n - 1] == T_EOS;
	const char *why = g.tool ? "tool_use" : eos ? "end_turn" : out >= maxn ? "max_tokens" : "end_turn";
	fprintf(stderr, "[air] prefill %d in %.2f s (%.0f tok/s) | %d tokens in %.2f s (%.1f tok/s) | %s%s | ctx %d\n", n - reused, t1 - t0,
		(n - reused) / (t1 - t0 + 1e-9), out, t2 - t1, out / (t2 - t1 + 1e-9), why, g.gone ? " (client gone)" : "", E.n);
	pthread_mutex_unlock(&mu);
	sputs(&g.content, "]");
	str e = { 0 };
	if (stream) {
		sprint(&e, "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"%s\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":%d}}", why, out);
		sse(&g, "message_delta", e.p, e.n); sse(&g, "message_stop", "{\"type\":\"message_stop\"}", 23);
	} else if (!g.gone) {
		sprint(&e, "{\"id\":\"%s\",\"type\":\"message\",\"role\":\"assistant\",\"model\":", g.id); sjson(&e, model, strlen(model));
		sputs(&e, ",\"content\":"); sadd(&e, g.content.p, g.content.n);
		sprint(&e, ",\"stop_reason\":\"%s\",\"stop_sequence\":null,\"usage\":{\"input_tokens\":%d,\"cache_read_input_tokens\":%d,\"cache_creation_input_tokens\":0,\"output_tokens\":%d}}", why, n - reused, reused, out);
		respond(fd, 200, "application/json", e.p, e.n);
	}
	sfree(&e); sfree(&g.name); sfree(&g.pend); sfree(&g.json); sfree(&g.content); sfree(&g.key); sfree(&g.cur); sfree(&g.tname); free(g.gen.p); free(t.p);
	return stream || g.gone;
}
static void *conn(void *arg) {
	int fd = (int)(intptr_t)arg, one = 1; str in = { 0 };
	setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
	for (;;) {
		char *he, b[65536]; ssize_t k;
		while (!(he = in.p ? strstr(in.p, "\r\n\r\n") : NULL)) { if ((k = read(fd, b, sizeof b)) <= 0 || in.n > (1 << 20)) goto done; sadd(&in, b, k); }
		size_t hl = he - in.p + 4; long cl = 0; char meth[16] = "", path[512] = "";
		sscanf(in.p, "%15s %511s", meth, path);
		for (char *l = strstr(in.p, "\r\n"); l && l < he; l = strstr(l + 2, "\r\n")) if (!strncasecmp(l + 2, "content-length:", 15)) cl = atol(l + 17);
		if (cl < 0 || cl > (256L << 20)) goto done;
		while (in.n < hl + cl) { if ((k = read(fd, b, sizeof b)) <= 0) goto done; sadd(&in, b, k); }
		char *q = strchr(path, '?'); if (q) *q = 0;
		int close_ = 0;
		if (!strcmp(meth, "GET") && (!strcmp(path, "/v1/models") || !strcmp(path, "/models"))) {
			str r = { 0 }; sprint(&r, "{\"data\":[{\"type\":\"model\",\"id\":\"%s\",\"display_name\":\"Inkling 2 Mini\",\"created_at\":\"2026-01-01T00:00:00Z\"}],\"has_more\":false}", mname);
			respond(fd, 200, "application/json", r.p, r.n); sfree(&r);
		} else if (!strcmp(meth, "POST") && !strncmp(path, "/v1/messages", 12)) {
			J *rq = jparse(in.p + hl, cl);
			if (!rq) error(fd, 400, "invalid_request_error", "invalid JSON body");
			else if (!strcmp(path, "/v1/messages/count_tokens")) {
				ivec t = { 0 }; int bnd[3]; char err[256], r[64];
				pthread_mutex_lock(&mu); int bad = render(rq, &t, bnd, err, sizeof err); pthread_mutex_unlock(&mu);
				if (bad) error(fd, 400, "invalid_request_error", err); else respond(fd, 200, "application/json", r, snprintf(r, sizeof r, "{\"input_tokens\":%d}", t.n));
				free(t.p);
			} else if (!strcmp(path, "/v1/messages")) close_ = messages(fd, rq);
			else error(fd, 404, "not_found_error", "unknown endpoint");
			jfree(rq);
		} else if (!strcmp(meth, "GET") || !strcmp(meth, "HEAD")) { const char *r = "air: POST /v1/messages\n"; respond(fd, 200, "text/plain", r, strlen(r)); }
		else error(fd, 404, "not_found_error", "unknown endpoint");
		if (close_) break;
		memmove(in.p, in.p + hl + cl, in.n - hl - cl); in.n -= hl + cl; in.p[in.n] = 0;
	}
done:
	sfree(&in); close(fd); return NULL;
}
static int serve(const char *host, int port) {
	int s = socket(AF_INET, SOCK_STREAM, 0), one = 1; struct sockaddr_in a = { .sin_family = AF_INET, .sin_port = htons(port) };
	setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
	if (inet_pton(AF_INET, host, &a.sin_addr) != 1 || bind(s, (struct sockaddr *)&a, sizeof a) || listen(s, 16)) die("cannot listen on %s:%d", host, port);
	ck_index();
	fprintf(stderr, "listening on http://%s:%d (context %d)\n"
		"  ANTHROPIC_BASE_URL=http://%s:%d ANTHROPIC_AUTH_TOKEN=x ANTHROPIC_MODEL=%s CLAUDE_CODE_MAX_CONTEXT_TOKENS=%d claude\n",
		host, port, E.max_ctx, host, port, mname, E.max_ctx);
	for (;;) {
		int fd = accept(s, NULL, NULL); pthread_t t; pthread_attr_t at;
		if (fd < 0) continue;
		pthread_attr_init(&at); pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED); pthread_attr_setstacksize(&at, 8 << 20);
		if (pthread_create(&t, &at, conn, (void *)(intptr_t)fd)) close(fd);
		pthread_attr_destroy(&at);
	}
}

/* ================= terminal chat ================= */
static int show(void *c, int t) {
	int *think = c; const char *p; int n;
	if (t == T_THINK) { *think = 1; printf("\x1b[2m"); }
	else if (t == T_TEXT) { if (*think) printf("\x1b[0m\n"); *think = 0; }
	else if (t == T_EOS && *think) { printf("\x1b[0m\n"); *think = 0; }
	else if ((n = piece(t, &p)) > 0) fwrite(p, 1, n, stdout);
	fflush(stdout); return 0;
}
static int chat(void) {
	char line[1 << 16]; int first = 1;
	printf("air: Inkling 2 Mini. /reset clears the context, /quit exits.\n");
	for (;;) {
		printf("\n\x1b[1m> \x1b[0m"); fflush(stdout);
		if (!fgets(line, sizeof line, stdin)) break;
		line[strcspn(line, "\n")] = 0;
		if (!line[0]) continue;
		if (!strcmp(line, "/quit")) break;
		if (!strcmp(line, "/reset")) { E.n = 0; E.have = 0; first = 1; printf("(cleared)\n"); continue; }
		ivec t = { 0 };
		if (first) effort_msg(&t, effort0);
		msg(&t, T_USER, NULL, T_TEXT, line, strlen(line));
		if (E.n + t.n + 512 > E.max_ctx) { printf("(context full: /reset)\n"); free(t.p); continue; }
		first = 0;
		double t0 = now(); prefill(t.p, t.n);
		double t1 = now(); int think = 0, out = generate(E.max_ctx, show, &think);
		double t2 = now();
		if (think) printf("\x1b[0m");
		printf("\n\x1b[2m[%d tokens, %.1f tok/s | prefill %d in %.2f s | context %d]\x1b[0m\n", out, out / (t2 - t1), t.n, t1 - t0, E.n);
		free(t.p);
	}
	return 0;
}

static void usage(void) {
	fprintf(stderr, "usage: air prep <checkpoint dir> <out.air>\n"
		"       air chat [-m model.air] [-c ctx] [-e effort]\n"
		"       air serve [-m model.air] [-c ctx] [-e effort] [-h host] [-p port] [-d cachedir] [-H]\n");
	exit(1);
}
int main(int ac, char **av) {
	if (ac < 2) usage();
	if (!strcmp(av[1], "prep")) { if (ac != 4) usage(); return prep(av[2], av[3]); }
	const char *model = "inkling.air", *host = "127.0.0.1"; int port = 8080, ctx = 65536;
	for (int i = 2; i < ac; i++) {
		if (!strcmp(av[i], "-H")) { stubhaiku = 0; continue; }
		if (i + 1 >= ac) usage();
		if (!strcmp(av[i], "-m")) model = av[++i];
		else if (!strcmp(av[i], "-c")) ctx = atoi(av[++i]);
		else if (!strcmp(av[i], "-e")) effort0 = atof(av[++i]);
		else if (!strcmp(av[i], "-h")) host = av[++i];
		else if (!strcmp(av[i], "-p")) port = atoi(av[++i]);
		else if (!strcmp(av[i], "-d")) snprintf(ckdir, sizeof ckdir, "%s", av[++i]);
		else usage();
	}
	if (strcmp(av[1], "chat") && strcmp(av[1], "serve")) usage();
	signal(SIGPIPE, SIG_IGN);
	load(model);
	tok_init();
	E.max_ctx = ctx; E.hist = calloc(ctx + 64, sizeof(int));
	if (gpu_init(ctx)) return 1;
	return !strcmp(av[1], "chat") ? chat() : serve(host, port);
}
