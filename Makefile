CFLAGS = -O2 -std=gnu23 -mcpu=apple-m1 -Wall -Wno-unused-function

air: air.c gpu.m air.metal air.h
	cc $(CFLAGS) -o air air.c gpu.m -fobjc-arc -framework Metal -framework Foundation -licucore

clean:
	rm -f air
