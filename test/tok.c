/* tok.c - tokenize stdin with air's tokenizer: test/tok model.air < text */
#define main air_main
#include "../air.c"
#undef main
int main(int ac, char **av) {
	str s = { 0 }; char b[65536]; size_t n;
	while ((n = fread(b, 1, sizeof b, stdin)) > 0) sadd(&s, b, n);
	load(ac > 1 ? av[1] : "inkling.air"); tok_init();
	ivec t = { 0 }; tokenize(s.p ? s.p : "", s.n, &t);
	for (int i = 0; i < t.n; i++) printf("%d%c", t.p[i], i + 1 < t.n ? ' ' : '\n');
	return 0;
}
