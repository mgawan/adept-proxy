objects = main.o kernel.o driver.o utils.o alignments.o
# ARCH = compute_60
ifeq ($(DEBUG),TRUE)
	NVCCFLAGS = -g -G -Xcompiler -fopenmp
else
	NVCCFLAGS = -O3 -I$(ROCM_PATH)/include
endif

program_gpu: $(objects)
	hipcc -std=c++11 $(NVCCFLAGS) $(objects) -o program_gpu
main.o: main.cpp driver.hpp
	hipcc -std=c++11 $(NVCCFLAGS) -I. -c main.cpp -o $@
driver.o: driver.cpp driver.hpp utils.hpp alignments.hpp
	hipcc -std=c++11 $(NVCCFLAGS) -I. -c driver.cpp -o $@
kernel.o: kernel.cpp kernel.hpp
	hipcc -std=c++11 $(NVCCFLAGS) -I. -c kernel.cpp -o $@
utils.o: utils.cpp utils.hpp
	hipcc -std=c++11 $(NVCCFLAGS) -I. -c utils.cpp -o $@
alignments.o: alignments.cpp alignments.hpp utils.hpp
	hipcc -std=c++11 $(NVCCFLAGS) -I. -c alignments.cpp -o $@
clean:
	rm *.o program_gpu
	echo "all object and executables deleted"
