/* bench.c - prefill and decode speed: test/bench model.air tokens.txt [n_prefill...]
 * prefills the first n tokens of the file from scratch for each n, then decodes 256 tokens after the last one. */
#define main air_main
#include "../air.c"
#undef main
static int count(void *c, int t) { (void)c; (void)t; return 0; }
int main(int ac, char **av) {
	if (ac < 3) { fprintf(stderr, "usage: bench model.air tokens.txt [n...]\n"); return 1; }
	FILE *f = fopen(av[2], "r"); ivec t = { 0 }; int x;
	if (!f) die("cannot open %s", av[2]);
	while (fscanf(f, "%d", &x) == 1) ipush(&t, x);
	load(av[1]); tok_init();
	int mx = 0; for (int i = 3; i < ac; i++) if (atoi(av[i]) > mx) mx = atoi(av[i]);
	if (ac == 3 || mx > t.n) mx = t.n;
	if (mx > 65536) die("context too large");
	E.max_ctx = mx + 512; E.hist = calloc(E.max_ctx + 64, sizeof(int));
	if (gpu_init(E.max_ctx)) return 1;
	int n = t.n;
	for (int i = 3; i <= ac; i++) {
		n = i < ac ? atoi(av[i]) : n; if (n > t.n) n = t.n;
		if (i == ac && ac > 3) break;
		E.n = 0; E.have = 0;
		double t0 = now(); prefill(t.p, n); double t1 = now();
		printf("prefill %6d tokens: %7.2f s  %6.0f tok/s\n", n, t1 - t0, n / (t1 - t0)); fflush(stdout);
	}
	double t0 = now(); int m = generate(256, count, NULL); double t1 = now();
	printf("decode  %6d tokens: %7.2f s  %6.1f tok/s (context %d)\n", m, t1 - t0, m / (t1 - t0), E.n);
	return 0;
}
