#pragma once

#include <type_traits>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <assert.h>

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
    int vals[] = {args...};

    elements_ = 1;
    for (int i=0; i<N; i++) {
      dims_[i] = vals[i];
      elements_ *= dims_[i];

      BOOST_ASSERT(dims_[i] > 0);
    }

    cudaError_t err = cudaMalloc(&d_data_, sizeof(T) * elements_);
    BOOST_ASSERT(err == cudaSuccess);

    ref_count_ = (int*) malloc(sizeof(int));
    (*ref_count_) = 1;
  }

  __host__ GpuData(const GpuData<N, T> &d) {
    for (int i=0; i<N; i++) {
      dims_[i] = d.dims_[i];
    }
    elements_ = d.elements_;
    coalesce_dim_ = d.coalesce_dim_;

    d_data_ = d.d_data_;
    ref_count_ = d.ref_count_;
    AddRef();
  }

  __host__ ~GpuData() {
    Release();
  }

  void Clear() {
    cudaError_t err = cudaMemset(d_data_, 0, sizeof(T) * elements_);
    BOOST_ASSERT(err == cudaSuccess);
  }

  const T* GetDevicePointer() const {
    return d_data_;
  }

  T* GetDevicePointer() {
    return d_data_;
  }

  __host__ void CopyFrom(const std::vector<T> &data, int el=-1) {
    if (el < 0) {
      BOOST_ASSERT(data.size() == elements_);
      el = elements_;
    }

    BOOST_ASSERT(data.size() >= el);
    BOOST_ASSERT(elements_ >= el);

    cudaError_t err = cudaMemcpy(d_data_, data.data(), sizeof(T) * el, cudaMemcpyHostToDevice);
    BOOST_ASSERT(err == cudaSuccess);
  }

  __host__ void CopyFrom(const GpuData<N, T> &data, int el=-1) {
    if (el < 0) {
      BOOST_ASSERT(data.Size() == elements_);
      el = elements_;
    }

    BOOST_ASSERT(data.Size() >= el);
    BOOST_ASSERT(elements_ >= el);

    cudaError_t err = cudaMemcpy(d_data_, data.GetDevicePointer(), sizeof(T) * el, cudaMemcpyDeviceToDevice);
    BOOST_ASSERT(err == cudaSuccess);
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

  __host__ __device__ int GetCoalesceDim() const {
    return coalesce_dim_;
  }

  template<class...T2, typename std::enable_if<sizeof...(T2) == N, int>::type = 0>
  __device__ T& operator()(const T2... args) {
    int idx = GetIdx(args...);
    return d_data_[idx];
  }

  template<class...T2, typename std::enable_if<sizeof...(T2) == N, int>::type = 0>
  __device__ T operator()(const T2... args) const {
    int idx = GetIdx(args...);
    return d_data_[idx];
  }

  __host__ __device__ int GetDim(int x) const {
    assert(x >= 0 && x < N);

    return dims_[x];
  }

  __host__ __device__ int Size() const {
    return elements_;
  }

  __host__ GpuData<N, T>& operator=(const GpuData<N, T> &d) {
    if (this != & d) {

      // Get rid of old data
      Release();

      // Now get new data
      for (int i=0; i<N; i++) {
        dims_[i] = d.dims_[i];
      }
      elements_ = d.elements_;

      coalesce_dim_ = d.coalesce_dim_;

      d_data_ = d.d_data_;
      ref_count_ = d.ref_count_;
      AddRef();
    }

    return *this;
  }


 private:
  int dims_[N];
  int elements_ = 0;

  int coalesce_dim_ = N - 1;

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
    BOOST_ASSERT( (*ref_count_) > 0 );

    (*ref_count_)--;

    if ((*ref_count_) == 0) {
      Cleanup();
    }
  }

  template<class...T2, typename std::enable_if<sizeof...(T2) == N, int>::type = 0>
  __device__ inline int GetIdx(const T2... args) const {
    int vals[] = {args...};

    int idx = 0;

    for (int i=0; i<N; i++) {
      // check for coalescing settings
      if (i == coalesce_dim_) {
        continue;
      }

      // assert valid dim
      assert(vals[i] < dims_[i]);
      assert(vals[i] >= 0);

      // Compute idx
      idx *= dims_[i];
      idx += vals[i];
    }

    // Now handle coalescing
    assert(vals[coalesce_dim_] < dims_[coalesce_dim_]);
    assert(vals[coalesce_dim_] >= 0);

    idx *= dims_[coalesce_dim_];
    idx += vals[coalesce_dim_];

    return idx;
  }
};

} // namespace gpu_util
} // namespace library
