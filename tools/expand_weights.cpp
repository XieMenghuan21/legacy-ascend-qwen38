#include "llama.h"
#include <cstdio>
int main(int argc,char **argv) {
    if(argc!=3) {fprintf(stderr,"Usage: expand_weights quantized.gguf expanded.gguf\n");return 2;}
    auto params=llama_model_quantize_default_params();
    params.nthread=16;
    params.ftype=LLAMA_FTYPE_MOSTLY_F16;
    params.allow_requantize=true;
    return llama_model_quantize(argv[1],argv[2],&params);
}
