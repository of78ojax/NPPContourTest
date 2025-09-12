#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <string>


namespace util
{
	inline void FailAndPrint(const std::string msg)
	{
		std::cerr << msg << "\n";
		throw std::runtime_error(msg);
	}

	inline void checkCudaErrors(cudaError error)
	{
		if (error != cudaSuccess)
		{
			std::string errorMSG(cudaGetErrorString(error));
			FailAndPrint(errorMSG);
		}
	}

	inline void PrintWarning(const std::string msg)
	{
		std::cout << msg << "\n";
	}
}



class CUDABuffer
{
protected:
	size_t sizeInBytes{ 0 };
	void* dPtr{ nullptr };
public:
	CUDABuffer() = default;
	CUDABuffer(void* ptr, size_t sizeinBytes) : sizeInBytes(sizeinBytes), dPtr(ptr) {}

	// implicit conversion to
	CUdeviceptr d_pointer()
	{
		return reinterpret_cast<CUdeviceptr>(dPtr);
	}

	template<typename T>
	explicit operator T() {
		return static_cast<T>(dPtr);
	}

	void zero()
	{
		setTo((unsigned char)0);
	}

	template<typename T>
	void setTo(const T value)
	{
		if (dPtr)
		{
			if (sizeof(T) == 1) {
				util::checkCudaErrors((cudaError_t)
					cuMemsetD8(d_pointer(), (uint8_t)value, sizeInBytes));
			}
			else if (sizeof(T) == 2) {
				size_t n = sizeInBytes / sizeof(T);
				if (n * sizeof(T) != sizeInBytes)
					util::FailAndPrint("Buffer not aligned with given datatype!");

				util::checkCudaErrors((cudaError_t)
					cuMemsetD16(d_pointer(), (uint16_t)value, n));
			}
			else if (sizeof(T) == 4)
			{
				size_t n = sizeInBytes / sizeof(T);
				if (n * sizeof(T) != sizeInBytes)
					util::FailAndPrint("Buffer not aligned with given datatype!");

				util::checkCudaErrors((cudaError_t)
					cuMemsetD32(d_pointer(), (uint32_t)value, n));
			}
			else
			{
				util::FailAndPrint("What Datatype u wanna use????");
			}

		}
	}

	template<typename T>
	void upload(const T* t, size_t count)
	{
		if (!dPtr)
			util::FailAndPrint("Buffer not allocated!");
		if (sizeInBytes > count * sizeof(T))
			util::PrintWarning("Buffer uploading less memory than buffer provides!");
		if (sizeInBytes < count * sizeof(T))
			util::FailAndPrint("Buffer uploading more memory than buffer provides!");

		util::checkCudaErrors(cudaMemcpy(dPtr, (void*)t,
			count * sizeof(T), cudaMemcpyHostToDevice));
	}

	template<typename T>
	void download(T* t, size_t count)
	{
		if (!dPtr)
			util::FailAndPrint("Buffer not allocated!");
		if (sizeInBytes > count * sizeof(T))
			util::PrintWarning("Buffer downloading less memory than buffer provides!");
		if (sizeInBytes < count * sizeof(T))
			util::FailAndPrint("Buffer downloading more memory than buffer provides!");

		util::checkCudaErrors(cudaMemcpy((void*)t, dPtr,
			count * sizeof(T), cudaMemcpyDeviceToHost));
	}

	void* d_ptr() const
	{
		return dPtr;
	}

	size_t getSizeInBytes() const
	{
		return sizeInBytes;
	}

};


class ManagedCUDABuffer : public CUDABuffer
{
public:
	ManagedCUDABuffer() {};
	~ManagedCUDABuffer() {
		if (dPtr)
			free();
	}
	ManagedCUDABuffer(size_t size)
	{
		alloc(size);
	}

	//! re-size buffer to given number of bytes
	void resize(size_t size)
	{
		if (dPtr) free();
		alloc(size);
	}

	//! allocate to given number of bytes
	virtual void alloc(size_t size)
	{
		if (dPtr)
			util::FailAndPrint("Buffer already allocated!");
		//assert(size % 16ull == 0);
		this->sizeInBytes = size;
		util::checkCudaErrors(cudaMalloc((void**)&dPtr, sizeInBytes));

	}

	//! free allocated memory
	virtual void free()
	{
		util::checkCudaErrors(cudaFree(dPtr));
		dPtr = nullptr;
		sizeInBytes = 0;
	}

	template<typename T>
	void alloc_and_upload(const std::vector<T>& vt)
	{

		resize(vt.size() * sizeof(T));
		upload((const T*)vt.data(), vt.size());
	}



};

struct PinnedCUDABuffer : public CUDABuffer {
public:
	~PinnedCUDABuffer() {
		if (dPtr)
			free();
	}

	//! re-size buffer to given number of bytes
	void resize(size_t size)
	{
		if (dPtr) free();
		alloc(size);
	}

	//! allocate to given number of bytes
	void alloc(size_t size)
	{
		if (dPtr)
			util::FailAndPrint("Buffer not allocated!");
		//assert(size % 16ull == 0);
		this->sizeInBytes = size;
		util::checkCudaErrors(cudaMallocHost((void**)&dPtr, sizeInBytes));

	}

	//! free allocated memory
	void free()
	{
		util::checkCudaErrors(cudaFreeHost(dPtr));
		dPtr = nullptr;
		sizeInBytes = 0;
	}

	template<typename T>
	void alloc_and_upload(const std::vector<T>& vt)
	{
		alloc(vt.size() * sizeof(T));
		upload((const T*)vt.data(), vt.size());
	}

	template<typename T>
	void upload(const T* t, size_t count, cudaStream_t stream)
	{
		if (!dPtr)
			util::FailAndPrint("Buffer not allocated!");
		if (sizeInBytes != count * sizeof(T))
			util::FailAndPrint("Buffer size mismatch!");
		util::checkCudaErrors(cudaMemcpyAsync(dPtr, (void*)t,
			count * sizeof(T), cudaMemcpyHostToDevice, stream));
	}

	template<typename T>
	void download(T* t, size_t count, cudaStream_t stream)
	{
		if (!dPtr)
			util::FailAndPrint("Buffer not allocated!");
		if (sizeInBytes != count * sizeof(T))
			util::FailAndPrint("Buffer size mismatch!");
		util::checkCudaErrors(cudaMemcpyAsync((void*)t, dPtr,
			count * sizeof(T), cudaMemcpyDeviceToHost, stream));
	}

};
