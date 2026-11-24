CFLAGS = -O2 -std=gnu23 -mcpu=apple-m1 -Wno-unused-function
LIBS = ../gpu.m -fobjc-arc -framework Metal -framework Foundation -licucore

all: tok geval bench
tok: tok.c ../air.c ../gpu.m ../air.metal ../air.h
	cc $(CFLAGS) -o $@ tok.c $(LIBS)
geval: geval.c ../air.c ../gpu.m ../air.metal ../air.h
	cc $(CFLAGS) -o $@ geval.c $(LIBS)
clean:
	rm -f tok geval bench
bench: bench.c ../air.c ../gpu.m ../air.metal ../air.h
	cc $(CFLAGS) -o $@ bench.c $(LIBS)
