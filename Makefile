# Makefile untuk KDD Dataset Analysis

CXX = g++
CXXFLAGS = -std=c++11 -Wall -O2
LDFLAGS = -lm

# Target utama
all: kdd_analyzer probability_demo

# Program utama untuk analisis KDD
kdd_analyzer: kdd_analysis.cpp
	$(CXX) $(CXXFLAGS) -o kdd_analyzer kdd_analysis.cpp $(LDFLAGS)

# Program demo fungsi probabilitas
probability_demo: probability_functions.cpp
	$(CXX) $(CXXFLAGS) -o probability_demo probability_functions.cpp $(LDFLAGS)

# Clean
clean:
	rm -f kdd_analyzer probability_demo *.o

# Run main analysis
run: kdd_analyzer
	./kdd_analyzer

# Run probability demo
run_prob: probability_demo
	./probability_demo

# Instructions
help:
	@echo "=== KDD Dataset Analysis Makefile ==="
	@echo "Perintah yang tersedia:"
	@echo "  make all          - Compile semua program"
	@echo "  make kdd_analyzer - Compile program analisis utama"
	@echo "  make probability_demo - Compile demo fungsi probabilitas"
	@echo "  make run          - Jalankan analisis KDD"
	@echo "  make run_prob     - Jalankan demo probabilitas"
	@echo "  make clean        - Hapus file executable"
	@echo ""
	@echo "Pastikan file KDDTrain.arff dan KDDTest.arff ada di direktori yang sama"

.PHONY: all clean run run_prob help