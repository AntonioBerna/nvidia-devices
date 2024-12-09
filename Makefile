CC=nvcc
CFLAGS=-O3
TARGET=nvidia-devices
SRCS=src/nvidia-devices.cu

all: $(SRCS)
	@mkdir -p bin
	$(CC) $(CFLAGS) -o bin/$(TARGET) $(SRCS)

.PHONY: clean

clean:
	$(RM) -r bin/