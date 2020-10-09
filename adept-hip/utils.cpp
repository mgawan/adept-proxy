#include<utils.hpp>
unsigned getMaxLength (std::vector<std::string> v)
{
  unsigned maxLength = 0;
  for(auto str : v){
    if(maxLength < str.length()){
      maxLength = str.length();
    }
  }
  return maxLength;
}

void initialize_alignments(gpu_bsw_driver::alignment_results *alignments, int max_alignments){
    hipHostMalloc(&(alignments->ref_begin), sizeof(short)*max_alignments);
    hipHostMalloc(&(alignments->ref_end), sizeof(short)*max_alignments);
    hipHostMalloc(&(alignments->query_begin), sizeof(short)*max_alignments);
    hipHostMalloc(&(alignments->query_end), sizeof(short)*max_alignments);
    hipHostMalloc(&(alignments->top_scores), sizeof(short)*max_alignments);
}

void free_alignments(gpu_bsw_driver::alignment_results *alignments){
       CHECK(hipHostFree(alignments->ref_begin));
       CHECK(hipHostFree(alignments->ref_end));
       CHECK(hipHostFree(alignments->query_begin));
       CHECK(hipHostFree(alignments->query_end));
       CHECK(hipHostFree(alignments->top_scores));

}

void asynch_mem_copies_htd(gpu_alignments* gpu_data, unsigned* offsetA_h, unsigned* offsetB_h, char* strA, char* strA_d, char* strB, char* strB_d, unsigned half_length_A, 
unsigned half_length_B, unsigned totalLengthA, unsigned totalLengthB, int sequences_per_stream, int sequences_stream_leftover, hipStream_t* streams_cuda){

        CHECK(hipMemcpyAsync(gpu_data->offset_ref_gpu, offsetA_h, (sequences_per_stream) * sizeof(int),
        hipMemcpyHostToDevice,streams_cuda[0]));
        CHECK(hipMemcpyAsync(gpu_data->offset_ref_gpu + sequences_per_stream, offsetA_h + sequences_per_stream, 
        (sequences_per_stream + sequences_stream_leftover) * sizeof(int), hipMemcpyHostToDevice,streams_cuda[1]));

        CHECK(hipMemcpyAsync(gpu_data->offset_query_gpu, offsetB_h, (sequences_per_stream) * sizeof(int),
        hipMemcpyHostToDevice,streams_cuda[0]));
        CHECK(hipMemcpyAsync(gpu_data->offset_query_gpu + sequences_per_stream, offsetB_h + sequences_per_stream, 
        (sequences_per_stream + sequences_stream_leftover) * sizeof(int), hipMemcpyHostToDevice,streams_cuda[1]));


        CHECK(hipMemcpyAsync(strA_d, strA, half_length_A * sizeof(char),
                              hipMemcpyHostToDevice,streams_cuda[0]));
        CHECK(hipMemcpyAsync(strA_d + half_length_A, strA + half_length_A, (totalLengthA - half_length_A) * sizeof(char),
                              hipMemcpyHostToDevice,streams_cuda[1]));

        CHECK(hipMemcpyAsync(strB_d, strB, half_length_B * sizeof(char),
                              hipMemcpyHostToDevice,streams_cuda[0]));
        CHECK(hipMemcpyAsync(strB_d + half_length_B, strB + half_length_B, (totalLengthB - half_length_B) * sizeof(char),
                              hipMemcpyHostToDevice,streams_cuda[1]));

}

void asynch_mem_copies_dth_mid(gpu_alignments* gpu_data, short* alAend, short* alBend, int sequences_per_stream, int sequences_stream_leftover, hipStream_t* streams_cuda){
            CHECK(hipMemcpyAsync(alAend, gpu_data->ref_end_gpu, sequences_per_stream * sizeof(short),
                hipMemcpyDeviceToHost, streams_cuda[0]));
            CHECK(hipMemcpyAsync(alAend + sequences_per_stream, gpu_data->ref_end_gpu + sequences_per_stream, 
                (sequences_per_stream + sequences_stream_leftover) * sizeof(short), hipMemcpyDeviceToHost, streams_cuda[1]));

            CHECK(hipMemcpyAsync(alBend, gpu_data->query_end_gpu, sequences_per_stream * sizeof(short), hipMemcpyDeviceToHost, streams_cuda[0]));
            CHECK(hipMemcpyAsync(alBend + sequences_per_stream, gpu_data->query_end_gpu + sequences_per_stream, (sequences_per_stream + sequences_stream_leftover) * sizeof(short), 
                hipMemcpyDeviceToHost, streams_cuda[1]));
}

void asynch_mem_copies_dth(gpu_alignments* gpu_data, short* alAbeg, short* alBbeg, short* top_scores_cpu, int sequences_per_stream, int sequences_stream_leftover, hipStream_t* streams_cuda){
           CHECK(hipMemcpyAsync(alAbeg, gpu_data->ref_start_gpu, sequences_per_stream * sizeof(short),
                                  hipMemcpyDeviceToHost, streams_cuda[0]));
          CHECK(hipMemcpyAsync(alAbeg + sequences_per_stream, gpu_data->ref_start_gpu + sequences_per_stream, (sequences_per_stream + sequences_stream_leftover) * sizeof(short),
                                  hipMemcpyDeviceToHost, streams_cuda[1]));

          CHECK(hipMemcpyAsync(alBbeg, gpu_data->query_start_gpu, sequences_per_stream * sizeof(short),
                          hipMemcpyDeviceToHost, streams_cuda[0]));
          CHECK(hipMemcpyAsync(alBbeg + sequences_per_stream, gpu_data->query_start_gpu + sequences_per_stream, (sequences_per_stream + sequences_stream_leftover) * sizeof(short),
                          hipMemcpyDeviceToHost, streams_cuda[1]));
                          
          CHECK(hipMemcpyAsync(top_scores_cpu, gpu_data->scores_gpu, sequences_per_stream * sizeof(short),
                          hipMemcpyDeviceToHost, streams_cuda[0]));
          CHECK(hipMemcpyAsync(top_scores_cpu + sequences_per_stream, gpu_data->scores_gpu + sequences_per_stream, 
          (sequences_per_stream + sequences_stream_leftover) * sizeof(short), hipMemcpyDeviceToHost, streams_cuda[1]));

}

int get_new_min_length(short* alAend, short* alBend, int blocksLaunched){
        int newMin = 1000;
        int maxA = 0;
        int maxB = 0;
        for(int i = 0; i < blocksLaunched; i++){
          if(alBend[i] > maxB ){
              maxB = alBend[i];
          }
          if(alAend[i] > maxA){
            maxA = alAend[i];
          }
        }
        newMin = (maxB > maxA)? maxA : maxB;
        return newMin;
}