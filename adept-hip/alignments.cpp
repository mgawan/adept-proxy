#include "alignments.hpp"
#include"utils.hpp"
gpu_alignments::gpu_alignments(int max_alignments){
    CHECK(hipMalloc(&offset_query_gpu, (max_alignments) * sizeof(int)));
    CHECK(hipMalloc(&offset_ref_gpu, (max_alignments) * sizeof(int)));
    CHECK(hipMalloc(&ref_start_gpu, (max_alignments) * sizeof(short)));
    CHECK(hipMalloc(&ref_end_gpu, (max_alignments) * sizeof(short)));
    CHECK(hipMalloc(&query_end_gpu, (max_alignments) * sizeof(short)));
    CHECK(hipMalloc(&query_start_gpu, (max_alignments) * sizeof(short)));
    CHECK(hipMalloc(&scores_gpu, (max_alignments) * sizeof(short)));
}

gpu_alignments::~gpu_alignments(){
    CHECK(hipFree(offset_ref_gpu));
    CHECK(hipFree(offset_query_gpu));
    CHECK(hipFree(ref_start_gpu));
    CHECK(hipFree(ref_end_gpu));
    CHECK(hipFree(query_start_gpu));
    CHECK(hipFree(query_end_gpu));
}