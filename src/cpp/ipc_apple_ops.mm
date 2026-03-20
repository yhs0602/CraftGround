#include <ATen/mps/MPSStream.h>
#include <torch/extension.h>
#import <Metal/Metal.h>
#include <pybind11/pybind11.h>
#include <algorithm>

namespace py = pybind11;

namespace {

struct NormalizeParams {
    uint32_t width;
    uint32_t height;
    uint32_t src_row_stride;
    uint32_t src_pixel_stride;
    uint32_t src_offset;
    uint32_t dst_row_stride;
    uint32_t dst_pixel_stride;
    uint32_t dst_offset;
};

id<MTLComputePipelineState> getNormalizePipeline(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipeline = nil;
    static dispatch_once_t once_token;
    static NSError *pipeline_error = nil;

    dispatch_once(&once_token, ^{
      NSString *source = [NSString stringWithUTF8String:R"(
        #include <metal_stdlib>
        using namespace metal;

        struct NormalizeParams {
            uint width;
            uint height;
            uint src_row_stride;
            uint src_pixel_stride;
            uint src_offset;
            uint dst_row_stride;
            uint dst_pixel_stride;
            uint dst_offset;
        };

        kernel void normalize_bgr_flip(
            device const uchar *src [[buffer(0)]],
            device uchar *dst [[buffer(1)]],
            constant NormalizeParams &params [[buffer(2)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            if (gid.x >= params.width || gid.y >= params.height) {
                return;
            }

            uint src_y = params.height - 1u - gid.y;
            uint src_base = params.src_offset + src_y * params.src_row_stride +
                gid.x * params.src_pixel_stride;
            uint dst_base = params.dst_offset + gid.y * params.dst_row_stride +
                gid.x * params.dst_pixel_stride;

            uchar b = src[src_base + 0];
            uchar g = src[src_base + 1];
            uchar r = src[src_base + 2];

            dst[dst_base + 0] = r;
            dst[dst_base + 1] = g;
            dst[dst_base + 2] = b;
        }
      )"];

      MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
      NSError *library_error = nil;
      id<MTLLibrary> library = [device newLibraryWithSource:source
                                                    options:options
                                                      error:&library_error];
      if (!library) {
          pipeline_error = library_error;
          return;
      }

      id<MTLFunction> function =
          [library newFunctionWithName:@"normalize_bgr_flip"];
      if (!function) {
          pipeline_error = [NSError
              errorWithDomain:@"craftground"
                         code:1
                     userInfo:@{
                         NSLocalizedDescriptionKey :
                             @"Failed to load normalize_bgr_flip kernel"
                     }];
          return;
      }

      NSError *compute_pipeline_error = nil;
      pipeline =
          [device newComputePipelineStateWithFunction:function
                                                error:&compute_pipeline_error];
      if (!pipeline) {
          pipeline_error = compute_pipeline_error;
      }
    });

    if (!pipeline) {
        std::string error_message = "Failed to build Metal normalize pipeline";
        if (pipeline_error && pipeline_error.localizedDescription) {
            error_message += ": ";
            error_message += [pipeline_error.localizedDescription UTF8String];
        }
        throw std::runtime_error(error_message);
    }

    return pipeline;
}

at::Tensor normalizeAppleTensor(const at::Tensor &src) {
    TORCH_CHECK(src.device().is_mps(), "Expected an MPS tensor");
    TORCH_CHECK(src.scalar_type() == at::kByte, "Expected a uint8 tensor");
    TORCH_CHECK(src.dim() == 3, "Expected an HWC tensor");
    TORCH_CHECK(src.size(2) == 3, "Expected a 3-channel BGR tensor");
    TORCH_CHECK(src.storage().data(), "Expected backing MTLBuffer storage");

    at::Tensor dst = at::empty({src.size(0), src.size(1), 3}, src.options());

    auto *stream = at::mps::getCurrentMPSStream();
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
    id<MTLComputePipelineState> pipeline =
        getNormalizePipeline(stream->device());
    id<MTLBuffer> src_buffer = (__bridge id<MTLBuffer>)src.storage().data();
    id<MTLBuffer> dst_buffer = (__bridge id<MTLBuffer>)dst.storage().data();

    NormalizeParams params{
        static_cast<uint32_t>(src.size(1)),
        static_cast<uint32_t>(src.size(0)),
        static_cast<uint32_t>(src.stride(0)),
        static_cast<uint32_t>(src.stride(1)),
        static_cast<uint32_t>(src.storage_offset()),
        static_cast<uint32_t>(dst.stride(0)),
        static_cast<uint32_t>(dst.stride(1)),
        static_cast<uint32_t>(dst.storage_offset()),
    };

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:src_buffer offset:0 atIndex:0];
    [encoder setBuffer:dst_buffer offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];

    NSUInteger thread_width = pipeline.threadExecutionWidth;
    NSUInteger thread_height = pipeline.maxTotalThreadsPerThreadgroup /
                               std::max<NSUInteger>(1, thread_width);
    if (thread_height == 0) {
        thread_height = 1;
    }

    MTLSize threads_per_threadgroup =
        MTLSizeMake(thread_width, thread_height, 1);
    MTLSize threads_per_grid = MTLSizeMake(
        static_cast<NSUInteger>(src.size(1)),
        static_cast<NSUInteger>(src.size(0)),
        1
    );
    [encoder dispatchThreads:threads_per_grid
        threadsPerThreadgroup:threads_per_threadgroup];

    return dst;
}

} // namespace

py::object normalize_apple_mtl_tensor_impl(py::object tensor) {
    return py::cast(normalizeAppleTensor(tensor.cast<at::Tensor>()));
}
