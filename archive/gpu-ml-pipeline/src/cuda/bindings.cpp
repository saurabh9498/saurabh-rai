/**
 * Python Bindings for CUDA Kernels
 * 
 * PyBind11 bindings exposing CUDA preprocessing kernels to Python.
 * Provides seamless integration with PyTorch tensors and NumPy arrays.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace py = pybind11;

// External declarations for CUDA kernels
extern "C" {
    void resize_bilinear_cuda(const uint8_t*, float*, int, int, int, int, int, cudaStream_t);
    void resize_nearest_cuda(const uint8_t*, uint8_t*, int, int, int, int, int, cudaStream_t);
    void normalize_imagenet_cuda(const uint8_t*, float*, int, int, int, cudaStream_t);
    void normalize_custom_cuda(const uint8_t*, float*, int, int, int, int, const float*, const float*, float, cudaStream_t);
    void fused_preprocess_cuda(const uint8_t*, float*, int, int, int, int, int, cudaStream_t);
    void fused_preprocess_fp16_cuda(const uint8_t*, void*, int, int, int, int, int, cudaStream_t);
    void fused_letterbox_cuda(const uint8_t*, float*, int, int, int, int, int, float, int, int, float, cudaStream_t);
    void nms_cuda(const float*, const float*, int64_t*, int*, int, int, float, cudaStream_t);
    void batched_nms_cuda(const float*, const float*, const int64_t*, int64_t*, int*, int, int, int, float, float, cudaStream_t);
}


/**
 * GPU Memory Manager for efficient allocation
 */
class GPUMemoryManager {
public:
    GPUMemoryManager() : allocated_bytes_(0) {}
    
    void* allocate(size_t bytes) {
        void* ptr;
        cudaMalloc(&ptr, bytes);
        allocated_bytes_ += bytes;
        return ptr;
    }
    
    void free(void* ptr, size_t bytes) {
        cudaFree(ptr);
        allocated_bytes_ -= bytes;
    }
    
    size_t allocated_bytes() const { return allocated_bytes_; }
    
private:
    size_t allocated_bytes_;
};


/**
 * CUDA Stream wrapper
 */
class CUDAStream {
public:
    CUDAStream() {
        cudaStreamCreate(&stream_);
    }
    
    ~CUDAStream() {
        cudaStreamDestroy(stream_);
    }
    
    void synchronize() {
        cudaStreamSynchronize(stream_);
    }
    
    cudaStream_t get() { return stream_; }
    
private:
    cudaStream_t stream_;
};


/**
 * Fused preprocessing function
 * Resizes and normalizes images in a single GPU kernel
 */
py::array_t<float> fused_preprocess(
    py::array_t<uint8_t> input,
    int dst_height,
    int dst_width,
    bool use_fp16 = false
) {
    auto buf = input.request();
    
    if (buf.ndim != 4) {
        throw std::runtime_error("Input must be 4D (NHWC)");
    }
    
    int batch_size = buf.shape[0];
    int src_height = buf.shape[1];
    int src_width = buf.shape[2];
    int channels = buf.shape[3];
    
    if (channels != 3) {
        throw std::runtime_error("Input must have 3 channels");
    }
    
    // Allocate GPU memory
    size_t input_size = batch_size * src_height * src_width * channels;
    size_t output_size = batch_size * 3 * dst_height * dst_width;
    
    uint8_t* d_input;
    float* d_output;
    
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size * sizeof(float));
    
    // Copy input to GPU
    cudaMemcpy(d_input, buf.ptr, input_size, cudaMemcpyHostToDevice);
    
    // Run fused preprocessing kernel
    fused_preprocess_cuda(
        d_input, d_output,
        batch_size, src_height, src_width,
        dst_height, dst_width,
        nullptr  // default stream
    );
    
    // Allocate output array
    auto result = py::array_t<float>({batch_size, 3, dst_height, dst_width});
    auto result_buf = result.request();
    
    // Copy result back to CPU
    cudaMemcpy(result_buf.ptr, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);
    
    return result;
}


/**
 * Resize images on GPU
 */
py::array_t<float> resize_bilinear(
    py::array_t<uint8_t> input,
    int dst_height,
    int dst_width
) {
    auto buf = input.request();
    
    int height = buf.shape[0];
    int width = buf.shape[1];
    int channels = (buf.ndim == 3) ? buf.shape[2] : 1;
    
    // Allocate GPU memory
    uint8_t* d_input;
    float* d_output;
    
    cudaMalloc(&d_input, height * width * channels);
    cudaMalloc(&d_output, dst_height * dst_width * channels * sizeof(float));
    
    cudaMemcpy(d_input, buf.ptr, height * width * channels, cudaMemcpyHostToDevice);
    
    resize_bilinear_cuda(
        d_input, d_output,
        height, width, dst_height, dst_width,
        channels, nullptr
    );
    
    std::vector<ssize_t> shape;
    if (buf.ndim == 3) {
        shape = {dst_height, dst_width, channels};
    } else {
        shape = {dst_height, dst_width};
    }
    
    auto result = py::array_t<float>(shape);
    auto result_buf = result.request();
    
    cudaMemcpy(result_buf.ptr, d_output, 
               dst_height * dst_width * channels * sizeof(float),
               cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return result;
}


/**
 * Normalize images on GPU
 */
py::array_t<float> normalize_imagenet(
    py::array_t<uint8_t> input
) {
    auto buf = input.request();
    
    if (buf.ndim != 4) {
        throw std::runtime_error("Input must be 4D (NHWC)");
    }
    
    int batch_size = buf.shape[0];
    int height = buf.shape[1];
    int width = buf.shape[2];
    
    size_t input_size = batch_size * height * width * 3;
    size_t output_size = batch_size * 3 * height * width;
    
    uint8_t* d_input;
    float* d_output;
    
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size * sizeof(float));
    
    cudaMemcpy(d_input, buf.ptr, input_size, cudaMemcpyHostToDevice);
    
    normalize_imagenet_cuda(d_input, d_output, batch_size, height, width, nullptr);
    
    auto result = py::array_t<float>({batch_size, 3, height, width});
    auto result_buf = result.request();
    
    cudaMemcpy(result_buf.ptr, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return result;
}


/**
 * NMS on GPU
 */
std::tuple<py::array_t<int64_t>, int> nms(
    py::array_t<float> boxes,
    py::array_t<float> scores,
    float iou_threshold,
    int max_output = 100
) {
    auto boxes_buf = boxes.request();
    auto scores_buf = scores.request();
    
    int num_boxes = boxes_buf.shape[0];
    
    float* d_boxes;
    float* d_scores;
    int64_t* d_keep;
    int* d_num_keep;
    
    cudaMalloc(&d_boxes, num_boxes * 4 * sizeof(float));
    cudaMalloc(&d_scores, num_boxes * sizeof(float));
    cudaMalloc(&d_keep, max_output * sizeof(int64_t));
    cudaMalloc(&d_num_keep, sizeof(int));
    
    cudaMemcpy(d_boxes, boxes_buf.ptr, num_boxes * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scores, scores_buf.ptr, num_boxes * sizeof(float), cudaMemcpyHostToDevice);
    
    nms_cuda(d_boxes, d_scores, d_keep, d_num_keep, num_boxes, max_output, iou_threshold, nullptr);
    
    int num_keep;
    cudaMemcpy(&num_keep, d_num_keep, sizeof(int), cudaMemcpyDeviceToHost);
    
    auto keep = py::array_t<int64_t>(num_keep);
    auto keep_buf = keep.request();
    
    cudaMemcpy(keep_buf.ptr, d_keep, num_keep * sizeof(int64_t), cudaMemcpyDeviceToHost);
    
    cudaFree(d_boxes);
    cudaFree(d_scores);
    cudaFree(d_keep);
    cudaFree(d_num_keep);
    
    return std::make_tuple(keep, num_keep);
}


/**
 * Get CUDA device properties
 */
py::dict get_device_info(int device_id = 0) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    
    py::dict info;
    info["name"] = std::string(prop.name);
    info["compute_capability"] = std::to_string(prop.major) + "." + std::to_string(prop.minor);
    info["total_memory_gb"] = prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);
    info["multiprocessor_count"] = prop.multiProcessorCount;
    info["max_threads_per_block"] = prop.maxThreadsPerBlock;
    info["warp_size"] = prop.warpSize;
    info["memory_clock_mhz"] = prop.memoryClockRate / 1000;
    info["memory_bus_width"] = prop.memoryBusWidth;
    
    return info;
}


// PyBind11 module definition
PYBIND11_MODULE(cuda_kernels, m) {
    m.doc() = "GPU-accelerated preprocessing kernels for ML inference";
    
    // Preprocessing functions
    m.def("fused_preprocess", &fused_preprocess,
          "Fused resize + normalize preprocessing",
          py::arg("input"),
          py::arg("dst_height"),
          py::arg("dst_width"),
          py::arg("use_fp16") = false);
    
    m.def("resize_bilinear", &resize_bilinear,
          "Bilinear resize on GPU",
          py::arg("input"),
          py::arg("dst_height"),
          py::arg("dst_width"));
    
    m.def("normalize_imagenet", &normalize_imagenet,
          "ImageNet normalization (HWC->NCHW) on GPU",
          py::arg("input"));
    
    m.def("nms", &nms,
          "Non-maximum suppression on GPU",
          py::arg("boxes"),
          py::arg("scores"),
          py::arg("iou_threshold"),
          py::arg("max_output") = 100);
    
    // Utility functions
    m.def("get_device_info", &get_device_info,
          "Get CUDA device information",
          py::arg("device_id") = 0);
    
    // Classes
    py::class_<CUDAStream>(m, "CUDAStream")
        .def(py::init<>())
        .def("synchronize", &CUDAStream::synchronize);
    
    py::class_<GPUMemoryManager>(m, "GPUMemoryManager")
        .def(py::init<>())
        .def("allocated_bytes", &GPUMemoryManager::allocated_bytes);
}
