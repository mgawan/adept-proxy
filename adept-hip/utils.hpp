#ifndef UTILS_HPP
#define UTILS_HPP
#include "hip/hip_runtime.h"
#include "driver.hpp"

// #define cudaErrchk(ans)                                                                  \
//     {                                                                                    \
//         gpuAssert((ans), __FILE__, __LINE__);                                            \
//     }



// inline void
// gpuAssert(hipError_t code, const char* file, int line, bool abort = true)
// {
//     if(code != hipSuccess)
//     {
//         fprintf(stderr, "GPUassert: %s %s %d cpu:%d\n", hipGetErrorString(code), file, line,omp_get_thread_num());
//         if(abort)
//             exit(code);
//     }
// }

#define CHECK(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
	  }\
}

unsigned getMaxLength (std::vector<std::string> v);
void initialize_alignments(gpu_bsw_driver::alignment_results *alignments, int max_alignments);
void free_alignments(gpu_bsw_driver::alignment_results *alignments);
void asynch_mem_copies_htd(gpu_alignments* gpu_data, unsigned* offsetA_h, unsigned* offsetB_h, char* strA, char* strA_d, char* strB, char* strB_d, unsigned half_length_A, unsigned half_length_B, unsigned totalLengthA, unsigned totalLengthB, int sequences_per_stream, int sequences_stream_leftover, hipStream_t* streams_cuda);
int get_new_min_length(short* alAend, short* alBend, int blocksLaunched);
void asynch_mem_copies_dth_mid(gpu_alignments* gpu_data, short* alAend, short* alBend, int sequences_per_stream, int sequences_stream_leftover, hipStream_t* streams_cuda);
void asynch_mem_copies_dth(gpu_alignments* gpu_data, short* alAbeg, short* alBbeg, short* top_scores_cpu, int sequences_per_stream, int sequences_stream_leftover, hipStream_t* streams_cuda);
#endif