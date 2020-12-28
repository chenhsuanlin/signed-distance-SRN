CUR_PATH = $(shell pwd)
VSCAN_PATH = O-CNN/virtual_scanner/cpp/libvirtualscanner
VSCAN_INCLUDE = $(VSCAN_PATH)/include/virtual_scanner
VSCAN_SRC = $(VSCAN_PATH)/src

CC = g++
CFLAGS = -Wall -g

main: parse.o points.o
	$(CC) $(CFLAGS) -o parse parse.o points.o
	rm *.o

parse.o: parse.cpp
	$(CC) $(CFLAGS) -I $(VSCAN_INCLUDE) -c parse.cpp
points.o: $(VSCAN_INCLUDE)/points.h $(VSCAN_SRC)/points.cpp
	cd $(VSCAN_PATH) && \
	$(CC) $(CFLAGS) -I include -c src/points.cpp && \
	mv points.o $(CUR_PATH)

clean:
	rm parse
