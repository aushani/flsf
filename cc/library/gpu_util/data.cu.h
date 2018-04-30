#pragma once

#include <type_traits>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <assert.h>

#include <boost/assert.hpp>
#include <thrust/device_ptr.h>

namespace library {
namespace gpu_util {

enum class DataLocation {
  ON_DEVICE,
  ON_HOST,
};

template<int N, class T, DataLocation L>
class Data {
  // For easy access
  template<int N2, class T2, DataLocation L2>
  friend class Data;

 public:
  template<class...T2, typename std::enable_if<sizeof...(T2) == N, int>::type = 0>
  __host__ Data(const T2... args) {
    int vals[] = {args...};

    elements_ = 1;
    for (int i=0; i<N; i++) {
      dims_[i] = vals[i];
      elements_ *= dims_[i];

      BOOST_ASSERT(dims_[i] > 0);
    }

    Initialize();
  }

  __host__ Data(const Data<N, T, L> &d) {
    for (int i=0; i<N; i++) {
      dims_[i] = d.dims_[i];
    }
    elements_ = d.elements_;
    coalesce_dim_ = d.coalesce_dim_;

    data_ = d.data_;
    ref_count_ = d.ref_count_;
    AddRef();
  }

  template<DataLocation L2, typename std::enable_if<L != L2, int>::type = 0>
  __host__ Data(const Data<N, T, L2> &d) {
    for (int i=0; i<N; i++) {
      dims_[i] = d.dims_[i];
    }
    elements_ = d.elements_;
    coalesce_dim_ = d.coalesce_dim_;

    Initialize();
    CopyFrom(d);
  }

  __host__ ~Data() {
    Release();
  }

  void Clear() {
    if (L == DataLocation::ON_DEVICE) {
      cudaError_t err = cudaMemset(data_, 0, sizeof(T) * elements_);
      BOOST_ASSERT(err == cudaSuccess);
    } else { // if L == DataLocation::ON_HOST
      memset(data_, 0, sizeof(T) * elements_);
    }
  }

  __host__ __device__ const T* GetRawPointer() const {
    return data_;
  }

  __host__ __device__ T* GetRawPointer() {
    return data_;
  }

  __host__ void CopyFrom(const std::vector<T> &data, int el=-1) {
    if (el < 0) {
      BOOST_ASSERT(data.size() == elements_);
      el = elements_;
    }

    BOOST_ASSERT(data.size() >= el);
    BOOST_ASSERT(elements_ >= el);

    cudaMemcpyKind kind = cudaMemcpyDefault;
    if (L == DataLocation::ON_DEVICE) {
      kind = cudaMemcpyHostToDevice;
    } else { // if L == DataLocation::ON_HOST
      kind = cudaMemcpyHostToHost;
    }

    cudaError_t err = cudaMemcpy(data_, data.data(), sizeof(T) * el, kind);
    BOOST_ASSERT(err == cudaSuccess);
  }

  template<DataLocation L2>
  __host__ void CopyFrom(const Data<N, T, L2> &data, int el=-1) {
    if (el < 0) {
      BOOST_ASSERT(data.Size() == elements_);
      el = elements_;
    }

    BOOST_ASSERT(data.GetCoalesceDim() == coalesce_dim_);
    BOOST_ASSERT(data.Size() >= el);
    BOOST_ASSERT(elements_ >= el);

    cudaMemcpyKind kind = cudaMemcpyDefault;

    if (L == DataLocation::ON_HOST && L2 == DataLocation::ON_HOST) {
      kind = cudaMemcpyHostToHost;
    }

    if (L == DataLocation::ON_DEVICE && L2 == DataLocation::ON_HOST) {
      kind = cudaMemcpyHostToDevice;
    }

    if (L == DataLocation::ON_HOST && L2 == DataLocation::ON_DEVICE) {
      kind = cudaMemcpyDeviceToHost;
    }

    if (L == DataLocation::ON_DEVICE && L2 == DataLocation::ON_DEVICE) {
      kind = cudaMemcpyDeviceToDevice;
    }

    cudaError_t err = cudaMemcpy(data_, data.GetRawPointer(), sizeof(T) * el, kind);
    BOOST_ASSERT(err == cudaSuccess);
  }

  template<DataLocation L2 = L, typename std::enable_if<L2 == DataLocation::ON_DEVICE, int>::type = 0>
  __host__ thrust::device_ptr<T> Begin() {
    return thrust::device_ptr<T>(data_);
  }

  template<DataLocation L2 = L, typename std::enable_if<L2 == DataLocation::ON_DEVICE, int>::type = 0>
  __host__ thrust::device_ptr<T> End() {
    return thrust::device_ptr<T>(data_) + Size();
  }

  __host__ void SetCoalesceDim(int dim) {
    BOOST_ASSERT(dim >= 0 && dim < N);
    coalesce_dim_ = dim;
  }

  __host__ __device__ int GetCoalesceDim() const {
    return coalesce_dim_;
  }

  template<class...T2, typename std::enable_if<sizeof...(T2) == N, int>::type = 0>
  __host__ __device__ bool InRange(const T2... args) const {
    int vals[] = {args...};

    for (int i=0; i<N; i++) {
      if (vals[i] >= dims_[i]) {
        return false;
      }

      if (vals[i] < 0) {
        return false;
      }
    }

    return true;
  }

  template<class...T2, typename std::enable_if<sizeof...(T2) == N && L == DataLocation::ON_HOST, int>::type = 0>
  __host__ inline T& operator()(const T2... args) {
    BOOST_ASSERT(InRange(args...));

    int idx = GetIdx(args...);
    return data_[idx];
  }

  template<class...T2, typename std::enable_if<sizeof...(T2) == N && L == DataLocation::ON_HOST, int>::type = 0>
  __host__ inline T operator()(const T2... args) const {
    BOOST_ASSERT(InRange(args...));

    int idx = GetIdx(args...);
    return data_[idx];
  }

  template<class...T2, typename std::enable_if<sizeof...(T2) == N && L == DataLocation::ON_DEVICE, int>::type = 0>
  __device__ inline T& operator()(const T2... args) {
    assert(InRange(args...));

    int idx = GetIdx(args...);
    return data_[idx];
  }

  template<class...T2, typename std::enable_if<sizeof...(T2) == N && L == DataLocation::ON_DEVICE, int>::type = 0>
  __device__ inline T operator()(const T2... args) const {
    assert(InRange(args...));

    int idx = GetIdx(args...);
    return data_[idx];
  }

  __host__ __device__ int GetDim(int x) const {
    assert(x >= 0 && x < N);

    return dims_[x];
  }

  __host__ __device__ int Size() const {
    return elements_;
  }

  __host__ Data<N, T, L>& operator=(const Data<N, T, L> &d) {
    // Check if this is the same
    if (this == &d) {
      return *this;
    }

    // Get rid of old data
    Release();

    // Get params
    for (int i=0; i<N; i++) {
      dims_[i] = d.dims_[i];
    }
    elements_ = d.elements_;

    coalesce_dim_ = d.coalesce_dim_;

    // Get new data
    data_ = d.data_;
    ref_count_ = d.ref_count_;
    AddRef();

    return *this;
  }

  template<DataLocation L2, typename std::enable_if<L != L2, int>::type = 0>
  __host__ Data<N, T, L>& operator=(const Data<N, T, L2> &d) {
    // Check if this is the same
    if (this == &d) {
      return *this;
    }

    // Get rid of old data
    Release();

    // Get params
    for (int i=0; i<N; i++) {
      dims_[i] = d.dims_[i];
    }
    elements_ = d.elements_;

    coalesce_dim_ = d.coalesce_dim_;

    // Get new data
    Initialize();
    CopyFrom(d);

    return *this;
  }

 private:
  int dims_[N] = {0,};
  int elements_ = 0;

  int coalesce_dim_ = N - 1;

  T *data_ = nullptr;
  int *ref_count_ = nullptr;

  __host__ void Initialize() {
    // Handle data memory
    if (L == DataLocation::ON_DEVICE) {
      cudaError_t err = cudaMalloc(&data_, sizeof(T) * elements_);
      BOOST_ASSERT(err == cudaSuccess);
    } else { // if L === DataLocation::ON_HOST
      data_ = (T*) malloc(sizeof(T) * elements_);
    }

    // Handle reference count
    ref_count_ = (int*) malloc(sizeof(int));
    (*ref_count_) = 1;
  }

  __host__ void Cleanup() {
    // Free device memory
    if (L == DataLocation::ON_DEVICE) {
      cudaFree(data_);
    } else { // if L == DataLocation::ON_HOST
      free(data_);
    }

    data_ = nullptr;

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
  __host__ __device__ inline int GetIdx(const T2... args) const {
    int vals[] = {args...};

    int idx = 0;

    for (int i=0; i<N; i++) {
      // check for coalescing settings
      if (i == coalesce_dim_) {
        continue;
      }

      // Compute idx
      idx *= dims_[i];
      idx += vals[i];
    }

    idx *= dims_[coalesce_dim_];
    idx += vals[coalesce_dim_];

    return idx;
  }
};

} // namespace gpu_util
} // namespace library
