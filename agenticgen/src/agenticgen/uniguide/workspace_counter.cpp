// Typed read-only adapter for PyTorch 2.9's BLAS workspace maps.
// Called only after the teacher executor has joined every GPU producer.
#include <c10/core/Allocator.h>
#include <map>
#include <tuple>

namespace at::cuda {
using WorkspaceMap = std::map<std::tuple<void*, void*>, c10::DataPtr>;
WorkspaceMap& cublas_handle_stream_to_workspace();
WorkspaceMap& cublaslt_handle_stream_to_workspace();
}

extern "C" size_t ug_blas_workspace_count() {
    return at::cuda::cublas_handle_stream_to_workspace().size();
}
extern "C" size_t ug_blaslt_workspace_count() {
    return at::cuda::cublaslt_handle_stream_to_workspace().size();
}
