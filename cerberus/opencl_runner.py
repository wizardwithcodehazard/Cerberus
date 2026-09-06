"""Zero-Dependency Native OpenCL Hardware Profiler & Execution Engine.

Directly binds to OpenCL.dll (AMD / NVIDIA / Intel GPU drivers) via ctypes to measure:
1. Exact Host-to-Device Memory Transfer Time (nanoseconds)
2. Exact GPU Compute Kernel Execution Time (nanoseconds)
3. Exact Device-to-Host Memory Transfer Time (nanoseconds)
"""

import ctypes
import math
from ctypes import (
    c_uint32, c_uint64, c_int32, c_void_p, c_size_t, c_char_p,
    byref, create_string_buffer, POINTER, cast
)
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

# OpenCL Constants
CL_SUCCESS = 0
CL_PLATFORM_NAME = 0x0902
CL_DEVICE_NAME = 0x102B
CL_DEVICE_TYPE_GPU = 0x00000004
CL_DEVICE_TYPE_ALL = 0xFFFFFFFF
CL_MEM_READ_WRITE = 1 << 0
CL_MEM_READ_ONLY = 1 << 2
CL_MEM_WRITE_ONLY = 1 << 1
CL_MEM_COPY_HOST_PTR = 1 << 5
CL_QUEUE_PROFILING_ENABLE = 1 << 1
CL_PROFILING_COMMAND_START = 0x1282
CL_PROFILING_COMMAND_END = 0x1283

@dataclass
class GPUExecutionProfile:
    device_name: str
    transfer_in_ms: float
    kernel_compute_ms: float
    transfer_out_ms: float
    total_gpu_ms: float

class OpenCLEngine:
    """Hardware executor interfacing OpenCL.dll directly without third-party dependencies."""

    def __init__(self):
        self.cl = None
        self.platform = None
        self.device = None
        self.device_name = "Unknown GPU"
        self.context = None
        self.queue = None
        self._init_opencl()

    def _init_opencl(self):
        try:
            self.cl = ctypes.windll.LoadLibrary("OpenCL.dll")
        except Exception:
            try:
                self.cl = ctypes.cdll.LoadLibrary("libOpenCL.so")
            except Exception as e:
                raise RuntimeError(f"Could not load OpenCL driver library: {e}")

        # Bind function signatures
        self.cl.clGetPlatformIDs.argtypes = [c_uint32, c_void_p, c_void_p]
        self.cl.clGetPlatformIDs.restype = c_int32

        self.cl.clGetPlatformInfo.argtypes = [c_void_p, c_uint32, c_size_t, c_void_p, c_void_p]
        self.cl.clGetPlatformInfo.restype = c_int32

        self.cl.clGetDeviceIDs.argtypes = [c_void_p, c_uint64, c_uint32, c_void_p, c_void_p]
        self.cl.clGetDeviceIDs.restype = c_int32

        self.cl.clGetDeviceInfo.argtypes = [c_void_p, c_uint32, c_size_t, c_void_p, c_void_p]
        self.cl.clGetDeviceInfo.restype = c_int32

        self.cl.clCreateContext.argtypes = [c_void_p, c_uint32, c_void_p, c_void_p, c_void_p, c_void_p]
        self.cl.clCreateContext.restype = c_void_p

        self.cl.clCreateCommandQueue.argtypes = [c_void_p, c_void_p, c_uint64, c_void_p]
        self.cl.clCreateCommandQueue.restype = c_void_p

        self.cl.clCreateBuffer.argtypes = [c_void_p, c_uint64, c_size_t, c_void_p, c_void_p]
        self.cl.clCreateBuffer.restype = c_void_p

        self.cl.clEnqueueWriteBuffer.argtypes = [
            c_void_p, c_void_p, c_uint32, c_size_t, c_size_t, c_void_p, c_uint32, c_void_p, c_void_p
        ]
        self.cl.clEnqueueWriteBuffer.restype = c_int32

        self.cl.clEnqueueReadBuffer.argtypes = [
            c_void_p, c_void_p, c_uint32, c_size_t, c_size_t, c_void_p, c_uint32, c_void_p, c_void_p
        ]
        self.cl.clEnqueueReadBuffer.restype = c_int32

        self.cl.clCreateProgramWithSource.argtypes = [c_void_p, c_uint32, c_void_p, c_void_p, c_void_p]
        self.cl.clCreateProgramWithSource.restype = c_void_p

        self.cl.clBuildProgram.argtypes = [c_void_p, c_uint32, c_void_p, c_char_p, c_void_p, c_void_p]
        self.cl.clBuildProgram.restype = c_int32

        self.cl.clCreateKernel.argtypes = [c_void_p, c_char_p, c_void_p]
        self.cl.clCreateKernel.restype = c_void_p

        self.cl.clSetKernelArg.argtypes = [c_void_p, c_uint32, c_size_t, c_void_p]
        self.cl.clSetKernelArg.restype = c_int32

        self.cl.clEnqueueNDRangeKernel.argtypes = [
            c_void_p, c_void_p, c_uint32, c_void_p, c_void_p, c_void_p, c_uint32, c_void_p, c_void_p
        ]
        self.cl.clEnqueueNDRangeKernel.restype = c_int32

        self.cl.clWaitForEvents.argtypes = [c_uint32, c_void_p]
        self.cl.clWaitForEvents.restype = c_int32

        self.cl.clGetEventProfilingInfo.argtypes = [c_void_p, c_uint32, c_size_t, c_void_p, c_void_p]
        self.cl.clGetEventProfilingInfo.restype = c_int32

        self.cl.clReleaseMemObject.argtypes = [c_void_p]
        self.cl.clReleaseKernel.argtypes = [c_void_p]
        self.cl.clReleaseProgram.argtypes = [c_void_p]
        self.cl.clReleaseEvent.argtypes = [c_void_p]

        # Enumerate platform and primary GPU
        num_platforms = c_uint32(0)
        self.cl.clGetPlatformIDs(0, None, byref(num_platforms))
        if num_platforms.value == 0:
            raise RuntimeError("No OpenCL platforms found.")

        platforms = (c_void_p * num_platforms.value)()
        self.cl.clGetPlatformIDs(num_platforms.value, platforms, None)
        self.platform = platforms[0]

        num_devices = c_uint32(0)
        err = self.cl.clGetDeviceIDs(self.platform, CL_DEVICE_TYPE_ALL, 0, None, byref(num_devices))
        if err != 0 or num_devices.value == 0:
            raise RuntimeError("No OpenCL devices found.")

        devices = (c_void_p * num_devices.value)()
        self.cl.clGetDeviceIDs(self.platform, CL_DEVICE_TYPE_ALL, num_devices.value, devices, None)
        self.device = devices[0]

        dev_buf = create_string_buffer(256)
        self.cl.clGetDeviceInfo(self.device, CL_DEVICE_NAME, 256, dev_buf, None)
        self.device_name = dev_buf.value.decode("utf-8", errors="ignore")

        # Create Context and Profiling Command Queue
        err_code = c_int32(0)
        device_array = (c_void_p * 1)(self.device)
        self.context = self.cl.clCreateContext(None, 1, device_array, None, None, byref(err_code))
        if err_code.value != 0:
            raise RuntimeError(f"clCreateContext failed with code {err_code.value}")

        self.queue = self.cl.clCreateCommandQueue(self.context, self.device, CL_QUEUE_PROFILING_ENABLE, byref(err_code))
        if err_code.value != 0:
            raise RuntimeError(f"clCreateCommandQueue failed with code {err_code.value}")

    def profile_kernel_1d(self, kernel_name: str, kernel_source: str, size: int) -> GPUExecutionProfile:
        """Profiles a 1D kernel on the physical GPU with exact nanosecond timing."""
        import numpy as np

        # Allocate host arrays
        a = np.ones(size, dtype=np.float32) * 1.5
        b = np.ones(size, dtype=np.float32) * 2.5
        c = np.zeros(size, dtype=np.float32)
        n_bytes = a.nbytes

        err_code = c_int32(0)
        d_a = c_void_p(self.cl.clCreateBuffer(self.context, CL_MEM_READ_ONLY, n_bytes, None, byref(err_code)))
        d_b = c_void_p(self.cl.clCreateBuffer(self.context, CL_MEM_READ_ONLY, n_bytes, None, byref(err_code)))
        d_c = c_void_p(self.cl.clCreateBuffer(self.context, CL_MEM_WRITE_ONLY, n_bytes, None, byref(err_code)))

        # Event handles
        evt_write1 = c_void_p()
        evt_write2 = c_void_p()
        evt_kernel = c_void_p()
        evt_read = c_void_p()

        # Enqueue writes to GPU
        self.cl.clEnqueueWriteBuffer(self.queue, d_a, 0, 0, n_bytes, a.ctypes.data, 0, None, byref(evt_write1))
        self.cl.clEnqueueWriteBuffer(self.queue, d_b, 0, 0, n_bytes, b.ctypes.data, 0, None, byref(evt_write2))

        # Compile OpenCL C source
        src_bytes = kernel_source.encode('utf-8')
        src_ptr = cast(c_char_p(src_bytes), c_void_p)
        src_len = c_size_t(len(src_bytes))

        program = self.cl.clCreateProgramWithSource(self.context, 1, byref(src_ptr), byref(src_len), byref(err_code))
        device_array = (c_void_p * 1)(self.device)
        self.cl.clBuildProgram(program, 1, device_array, None, None, None)

        kernel = self.cl.clCreateKernel(program, kernel_name.encode('utf-8'), byref(err_code))

        n_val = c_int32(size)
        self.cl.clSetKernelArg(kernel, 0, sizeof(c_void_p), byref(d_a))
        self.cl.clSetKernelArg(kernel, 1, sizeof(c_void_p), byref(d_b))
        self.cl.clSetKernelArg(kernel, 2, sizeof(c_void_p), byref(d_c))
        self.cl.clSetKernelArg(kernel, 3, sizeof(c_int32), byref(n_val))

        # Launch kernel
        global_work_size = (c_size_t * 1)(size)
        self.cl.clEnqueueNDRangeKernel(self.queue, kernel, 1, None, global_work_size, None, 0, None, byref(evt_kernel))

    def profile_kernel_2d(self, kernel_name: str, kernel_source: str, dim: int) -> GPUExecutionProfile:
        """Profiles a 2D/3D matrix kernel on the physical GPU with exact nanosecond timing."""
        import numpy as np

        # Allocate 2D matrix buffers
        n_elems = dim * dim
        a = np.ones(n_elems, dtype=np.float32) * 1.2
        b = np.ones(n_elems, dtype=np.float32) * 0.8
        c = np.zeros(n_elems, dtype=np.float32)
        n_bytes = a.nbytes

        err_code = c_int32(0)
        d_a = c_void_p(self.cl.clCreateBuffer(self.context, CL_MEM_READ_ONLY, n_bytes, None, byref(err_code)))
        d_b = c_void_p(self.cl.clCreateBuffer(self.context, CL_MEM_READ_ONLY, n_bytes, None, byref(err_code)))
        d_c = c_void_p(self.cl.clCreateBuffer(self.context, CL_MEM_READ_WRITE, n_bytes, None, byref(err_code)))

        # Event handles
        evt_write1 = c_void_p()
        evt_write2 = c_void_p()
        evt_kernel = c_void_p()
        evt_read = c_void_p()

        # Enqueue writes to GPU
        self.cl.clEnqueueWriteBuffer(self.queue, d_a, 0, 0, n_bytes, a.ctypes.data, 0, None, byref(evt_write1))
        self.cl.clEnqueueWriteBuffer(self.queue, d_b, 0, 0, n_bytes, b.ctypes.data, 0, None, byref(evt_write2))

        # Compile OpenCL C source
        src_bytes = kernel_source.encode('utf-8')
        src_ptr = cast(c_char_p(src_bytes), c_void_p)
        src_len = c_size_t(len(src_bytes))

        program = self.cl.clCreateProgramWithSource(self.context, 1, byref(src_ptr), byref(src_len), byref(err_code))
        device_array = (c_void_p * 1)(self.device)
        self.cl.clBuildProgram(program, 1, device_array, None, None, None)

        kernel = self.cl.clCreateKernel(program, kernel_name.encode('utf-8'), byref(err_code))

        n_val = c_int32(dim)
        self.cl.clSetKernelArg(kernel, 0, sizeof(c_void_p), byref(d_a))
        self.cl.clSetKernelArg(kernel, 1, sizeof(c_void_p), byref(d_b))
        self.cl.clSetKernelArg(kernel, 2, sizeof(c_void_p), byref(d_c))
        self.cl.clSetKernelArg(kernel, 3, sizeof(c_int32), byref(n_val))

        # Launch 2D grid: (dim, dim)
        global_work_size = (c_size_t * 2)(dim, dim)
        self.cl.clEnqueueNDRangeKernel(self.queue, kernel, 2, None, global_work_size, None, 0, None, byref(evt_kernel))

        # Read back results
        self.cl.clEnqueueReadBuffer(self.queue, d_c, 1, 0, n_bytes, c.ctypes.data, 0, None, byref(evt_read))

        # Query profiling nanoseconds
        t_write1 = self._get_event_duration_ms(evt_write1)
        t_write2 = self._get_event_duration_ms(evt_write2)
        t_kernel = self._get_event_duration_ms(evt_kernel)
        t_read = self._get_event_duration_ms(evt_read)

        # Cleanup
        self.cl.clReleaseMemObject(d_a)
        self.cl.clReleaseMemObject(d_b)
        self.cl.clReleaseMemObject(d_c)
        self.cl.clReleaseKernel(kernel)
        self.cl.clReleaseProgram(program)
        self.cl.clReleaseEvent(evt_write1)
        self.cl.clReleaseEvent(evt_write2)
        self.cl.clReleaseEvent(evt_kernel)
        self.cl.clReleaseEvent(evt_read)

        t_transfer_in = t_write1 + t_write2
        t_total = t_transfer_in + t_kernel + t_read

        return GPUExecutionProfile(
            device_name=self.device_name,
            transfer_in_ms=t_transfer_in,
            kernel_compute_ms=t_kernel,
            transfer_out_ms=t_read,
            total_gpu_ms=t_total
        )

    def _get_event_duration_ms(self, event_handle: c_void_p) -> float:
        start = c_uint64(0)
        end = c_uint64(0)
        self.cl.clGetEventProfilingInfo(event_handle, CL_PROFILING_COMMAND_START, sizeof(c_uint64), byref(start), None)
        self.cl.clGetEventProfilingInfo(event_handle, CL_PROFILING_COMMAND_END, sizeof(c_uint64), byref(end), None)
        nanoseconds = float(end.value - start.value)
        return max(0.0001, nanoseconds / 1e6)

def sizeof(obj):
    return ctypes.sizeof(obj)
