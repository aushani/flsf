#pragma once

#include <type_traits>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>

#include <boost/assert.hpp>
#include <thrust/device_ptr.h>

//#include "library/util/host_data.h"

//namespace ut = library::util;

namespace library {
namespace gpu_util {

template<int N, class T = float>
class GpuData {
 public:
  template<class...T2, typename std::enable_if<sizeof...(T2) == N, int>::type = 0>
  __host__ GpuData(const T2... args) {
    size_t vals[] = {args...};

    elements_ = 1;
    for (int i=0; i<N; i++) {
      dims_[i] = vals[i];
      elements_ *= dims_[i];
    }

    cudaMalloc(&d_data_, sizeof(T) * elements_);

    ref_count_ = (int*) malloc(sizeof(int));
    (*ref_count_) = 1;
  }

  __host__ GpuData(const GpuData<N> &d) {
    for (int i=0; i<N; i++) {
      dims_[i] = d.dims_[i];
    }
    elements_ = d.elements_;

    d_data_ = d.d_data_;
    coalesce_dim_ = d.coalesce_dim_;

    ref_count_ = d.ref_count_;
    AddRef();
  }

  __host__ ~GpuData() {
    Release();
  }

  const T* GetDevicePointer() const {
    return d_data_;
  }

  T* GetDevicePointer() {
    return d_data_;
  }

  __host__ void SetData(const std::vector<T> &data) {
    BOOST_ASSERT(data.size() == elements_);
    cudaMemcpy(d_data_, data.data(), sizeof(T)*elements_, cudaMemcpyHostToDevice);

    cudaError_t err = cudaDeviceSynchronize();
    BOOST_ASSERT(err = cudaSuccess);
  }

  __host__ void SetPartialData(const std::vector<T> &data) {
    BOOST_ASSERT(data.size() <= elements_);
    cudaMemcpy(d_data_, data.data(), sizeof(T)*data.size(), cudaMemcpyHostToDevice);

    cudaError_t err = cudaDeviceSynchronize();
    BOOST_ASSERT(err = cudaSuccess);
  }

  __host__ thrust::device_ptr<T> Begin() {
    return thrust::device_ptr<T>(d_data_);
  }

  __host__ thrust::device_ptr<T> End() {
    return thrust::device_ptr<T>(d_data_) + Size();
  }

  /*
  ut::HostData<N> GetHostData() const {
    std::vector<T> data(elements_, 0.0);
    cudaMemcpy(data.data(), d_data_, sizeof(T)*elements_, cudaMemcpyDeviceToHost);

    ut::HostData<N> host_data(data, coalesce_dim_, dims_);
    return host_data;
  }

  __host__ void CopyDataFrom(const ut::HostData<N> &data) const {
    BOOST_ASSERT(Size() == data.Size());
    BOOST_ASSERT(GetCoalesceDim() == data.GetCoalesceDim());

    for (int i=0; i<N; i++) {
      BOOST_ASSERT(GetDim(i) == data.GetDim(i));
    }

    cudaMemcpy(d_data_, data.GetDataPointer(), elements_*sizeof(float), cudaMemcpyHostToDevice);

    cudaError_t err = cudaDeviceSynchronize();
    BOOST_ASSERT(err == cudaSuccess);
  }
  */

  __host__ void SetCoalesceDim(int dim) {
    BOOST_ASSERT(dim >= 0 && dim < N);
    coalesce_dim_ = dim;
  }

  __host__ int GetCoalesceDim() const {
    return coalesce_dim_;
  }

  template<class...T2, typename std::enable_if<sizeof...(T2) == N, int>::type = 0>
  __device__ T& operator()(const T2... args) {
    size_t idx = GetIdx(args...);
    return d_data_[idx];
  }

  template<class...T2, typename std::enable_if<sizeof...(T2) == N, int>::type = 0>
  __device__ T operator()(const T2... args) const {
    size_t idx = GetIdx(args...);
    return d_data_[idx];
  }

  __host__ __device__ int GetDim(int x) const {
    if (x < 0 || x >= N) {
      return -1;
    }

    return dims_[x];
  }

  __host__ __device__ size_t Size() const {
    return elements_;
  }

  // Load from file
  /*
  __host__ void LoadFrom(const std::string &path) {
    std::vector<float> data;

    std::ifstream file(path);

    float val;
    while (file >> val) {
      //printf("Read %5.3f\n", val);
      //val = 1.0;
      data.push_back(val);
    }

    //printf("Read %ld elements\n", data.size());

    BOOST_ASSERT(data.size() == elements_);

    cudaMemcpy(d_data_, data.data(), sizeof(float)*elements_, cudaMemcpyHostToDevice);
  }
  */

  __host__ GpuData<N, T>& operator=(const GpuData<N, T> &d) {
    if (this != & d) {

      // Get rid of old data
      Release();

      // Now get new data
      for (int i=0; i<N; i++) {
        dims_[i] = d.dims_[i];
      }

      d_data_ = d.d_data_;
      ref_count_ = d.ref_count_;
      AddRef();
    }

    return *this;
  }

 private:
  int dims_[N];
  size_t elements_ = 0;

  int coalesce_dim_ = -1;

  T *d_data_ = nullptr;
  int *ref_count_ = nullptr;

  __host__ void Cleanup() {
    // Free device memory
    cudaFree(d_data_);
    d_data_ = nullptr;

    // Free reference count
    free(ref_count_);
    ref_count_ = nullptr;
  }

  __host__ void AddRef() {
    (*ref_count_)++;
  }

  __host__ void Release() {
    (*ref_count_)--;

    if ((*ref_count_) == 0) {
      Cleanup();
    }
  }

  template<class...T2, typename std::enable_if<sizeof...(T2) == N, int>::type = 0>
  __device__ size_t GetIdx(const T2... args) const {
    size_t vals[] = {args...};

    size_t idx = 0;

    for (int i=0; i<N; i++) {
      // check for coalescing settings
      if (i == coalesce_dim_) {
        continue;
      }

      // assert valid dim
      idx *= dims_[i];
      idx += vals[i];
    }

    if (coalesce_dim_ >= 0) {
      idx *= dims_[coalesce_dim_];
      idx += vals[coalesce_dim_];
    }

    return idx;
  }
};

} // namespace gpu_util
} // namespace library
