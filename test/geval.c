/* geval.c - teacher-forced top-1 agreement with a reference: test/geval model.air tok.txt top.bin [prefill_n]
 * tok.txt: token ids; top.bin per position: f32 logsumexp, 64 x (i32 id, f32 logit), best first (llama.cpp ref_dump). */
#define main air_main
#include "../air.c"
#undef main
int main(int ac, char **av) {
	if (ac < 4) { fprintf(stderr, "usage: geval model.air tok.txt top.bin [prefill_n]\n"); return 1; }
	FILE *ft = fopen(av[2], "r"), *fr = fopen(av[3], "rb"); ivec t = { 0 }; int x;
	if (!ft || !fr) die("cannot open inputs");
	while (fscanf(ft, "%d", &x) == 1) ipush(&t, x);
	load(av[1]); tok_init();
	E.max_ctx = t.n + 64; E.hist = calloc(E.max_ctx + 64, sizeof(int));
	if (gpu_init(E.max_ctx)) return 1;
	int i0 = ac > 4 ? atoi(av[4]) : 0, agree = 0, cnt = 0, best[MAXB];
	struct { float lse; struct { int32_t id; float v; } c[64]; } r;
	if (i0 > 0) {		/* first i0 tokens through the prefill path: only its last row is compared */
		prefill(t.p, i0);
		for (int k = 0; k < i0; k++) if (fread(&r, sizeof r, 1, fr) != 1) die("short reference");
		agree += E.last == r.c[0].id; cnt++; E.have = 0;
	}
	double t0 = now();
	for (int i = i0; i < t.n; i += 3) {
		int b = t.n - i < 3 ? t.n - i : 3;
		fwd(t.p + i, b, 0, best, NULL);
		for (int k = 0; k < b; k++) { if (fread(&r, sizeof r, 1, fr) != 1) goto done; agree += best[k] == r.c[0].id; cnt++; }
	}
done:
	printf("%s: n=%d top1=%.4f (%.1f ms/forward)\n", av[2], cnt, (double)agree / cnt, (now() - t0) * 1e3 / ((t.n - i0 + 2) / 3));
	return 0;
}
