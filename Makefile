CXX := clang++

TARGET := mtflops
BUILD_DIR := build
OBJS := $(BUILD_DIR)/main.o $(BUILD_DIR)/gpu_bench.o

CXXFLAGS_COMMON := -O3 -std=c++20 -DNDEBUG -Wall -Wextra -Wpedantic -DACCELERATE_NEW_LAPACK
CXXFLAGS_CPP := $(CXXFLAGS_COMMON)
CXXFLAGS_MM := $(CXXFLAGS_COMMON) -fobjc-arc

LDFLAGS := -framework Accelerate -framework Foundation -framework Metal -framework QuartzCore

.PHONY: all clean

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/main.o: main.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_CPP) -c $< -o $@

$(BUILD_DIR)/gpu_bench.o: gpu_bench.mm gpu_bench.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_MM) -c $< -o $@

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)
	rm -rf $(BUILD_DIR)
