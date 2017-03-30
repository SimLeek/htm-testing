import numpy as np
import pyopencl as cl
import pyopencl.tools
import pyopencl.array


def ensure_required_version():
    ver = cl.get_cl_header_version()
    assert ver[0] >= 2 and ver[1] >= 0, "OpenCL must be version 2.0 or greater."


def check_number_of_GPUs():
    CL_DEVICE_TYPE_GPU = 4
    count = 0
    for p in cl.get_platforms():
        for d in p.get_devices():
            if d.get_info(cl.device_info.TYPE) & CL_DEVICE_TYPE_GPU == CL_DEVICE_TYPE_GPU:
                count += 1
    return count


def check_number_of_CPUs():
    CL_DEVICE_TYPE_CPU = 2
    count = 0
    for p in cl.get_platforms():
        for d in p.get_devices():
            if d.get_info(cl.device_info.TYPE) & CL_DEVICE_TYPE_CPU == CL_DEVICE_TYPE_CPU:
                count += 1
    return count


def get_too_much_info():
    for p in cl.get_platforms():
        print("platform:")
        print(p.get_info(cl.platform_info.NAME))
        print(p.get_info(cl.platform_info.PROFILE))
        print(p.get_info(cl.platform_info.VENDOR))
        print(p.get_info(cl.platform_info.VERSION))
        print(p.get_info(cl.platform_info.EXTENSIONS))
        for d in p.get_devices():
            print("\ndevice:")
            print(d.get_info(cl.device_info.ADDRESS_BITS))
            print(d.get_info(cl.device_info.AVAILABLE))
            print(d.get_info(cl.device_info.COMPILER_AVAILABLE))
            print(d.get_info(cl.device_info.EXECUTION_CAPABILITIES))
            print(d.get_info(cl.device_info.EXTENSIONS))
            print(d.get_info(cl.device_info.GLOBAL_MEM_CACHELINE_SIZE))
            print(d.get_info(cl.device_info.GLOBAL_MEM_CACHE_SIZE))
            print(d.get_info(cl.device_info.GLOBAL_MEM_CACHE_TYPE))
            print(d.get_info(cl.device_info.GLOBAL_MEM_SIZE))
            print(d.get_info(cl.device_info.IMAGE_SUPPORT))
            print(d.get_info(cl.device_info.LOCAL_MEM_SIZE))
            print(d.get_info(cl.device_info.LOCAL_MEM_TYPE))
            print(d.get_info(cl.device_info.MAX_CLOCK_FREQUENCY))
            print(d.get_info(cl.device_info.MAX_COMPUTE_UNITS))
            print(d.get_info(cl.device_info.MAX_CONSTANT_ARGS))
            print(d.get_info(cl.device_info.MAX_CONSTANT_BUFFER_SIZE))
            print(d.get_info(cl.device_info.MAX_WORK_GROUP_SIZE))
            print(d.get_info(cl.device_info.MAX_WORK_ITEM_DIMENSIONS))
            print(d.get_info(cl.device_info.MAX_WORK_ITEM_SIZES))
            print(d.get_info(cl.device_info.NAME))
            print(d.get_info(cl.device_info.OPENCL_C_VERSION))
            print(d.get_info(cl.device_info.PLATFORM))
            print(d.get_info(cl.device_info.PROFILE))
            print(d.get_info(cl.device_info.QUEUE_PROPERTIES))
            print("type", d.get_info(cl.device_info.TYPE))
            print(d.get_info(cl.device_info.VENDOR))
            print(d.get_info(cl.device_info.VENDOR_ID))
            print(d.get_info(cl.device_info.VERSION))
            try:
                print(d.get_info(cl.device_info.BUILT_IN_KERNELS))
            except RuntimeError:
                pass

            print('\n')


def get_registers_per_block(d):
    # get number of 32-bit registers available to a work group
    if "cl_nv_device_attribute_query" in d.get_info(cl.device_info.EXTENSIONS):
        return d.get_info(cl.device_info.REGISTERS_PER_BLOCK_NV)


def does_exec_timeout(d):
    # if this is 1, then GPU will eventually time out the persistent kernels...
    if "cl_nv_device_attribute_query" in d.get_info(cl.device_info.EXTENSIONS):
        if d.get_info(cl.device_info.KERNEL_EXEC_TIMEOUT_NV) == 1:
            return True
    return False


def does_gpu_overlap(d):
    if "cl_nv_device_attribute_query" in d.get_info(cl.device_info.EXTENSIONS):
        # if this is 1, then GPU overlaps, and memory can be copied while kernel is running
        # kind of important for synchronizing warps, or groups
        if d.get_info(cl.device_info.GPU_OVERLAP_NV) == 1:
            return True

    return False


def get_warp_size(d):  # I should rename this to get_subgroup_size
    if "cl_nv_device_attribute_query" in d.get_info(cl.device_info.EXTENSIONS):
        return d.get_info(cl.device_info.WARP_SIZE_NV)
    return -1


def does_byte_addressable_store(d):
    # put '#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable' in your .cl file if you want to use byte addressable storage
    if "cl_khr_byte_addressable_store" in d.get_info(cl.device_info.EXTENSIONS):
        return True
    return False


# in case other atomic function are needed:
# cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics


def manually_select_device():
    cl_selected_context = cl.create_some_context(interactive=True)
    return cl_selected_context


def allow_interactive_device_selection_again():
    import os
    del os.environ["PYOPENCL_CTX"]


def select_GPU():
    if check_number_of_GPUs() == 0:
        print("Sadly, you don't have any OpenCL ready GPUs.")
        exit()
    elif check_number_of_GPUs() > 1:
        print("You have " + str(check_number_of_GPUs()) + " GPUs. Please select which GPU you wish to use.")
        return manually_select_device()
    else:
        return cl.Context(dev_type=cl.device_type.GPU)


# Fortunately, my GPU doesn't seem to have a watchdog timer on it,
# but some people can't run kernels for longer than 5 seconds.
def display_watchdog_instructions():
    print("Your Operating System's watchdog timer is stopping HTM persistent threads.")
    print("If you are unable to turn off your GPU's watchdog timer, "
          "consider running from the Linux command line interface, or, use a non Nvidia GPU.")


a_np = np.random.rand(100, 100).astype(np.float32)
b_np = np.random.rand(100, 100).astype(np.float32)

cl_context = select_GPU()
cl_queue = cl.CommandQueue(cl_context)

cl_mem_flags = cl.mem_flags
#ALLOC_HOST_PTR needed for async xfer. Copy probably works too.


def check_struct_alignment(the_struct):
    my_struct, my_struct_c_decl = cl.tools.match_dtype_to_c_struct(cl_context.devices[0],
                                                                   "the_struct",
                                                                   the_struct)
    print(my_struct_c_decl)

from enum import Enum


class ControlCodes(Enum):
    REQUEST_END = 1 << 0

control_code = np.uint32(0)

persistent_thread_control_struct = np.dtype([("time_before_restart", np.int32), ("codes", np.uint32)])

#check_struct_alignment(persistent_thread_control_struct)

persistent_thread_control_struct = cl.tools.get_or_register_dtype("persistent_thread_control_struct",
                                                                  persistent_thread_control_struct)

host_control_struct = np.empty(100,persistent_thread_control_struct)
persistent_thread_control_struct["time_before_restart"].fill(120000)#for my machine
persistent_thread_control_struct["codes"].fill(control_code)#for my machine

sparse_output = np.zeros(100).astype(np.int32)

#// Get mapped pointers to pinned input host buffers
#    //   Note:  This allows general (non-OpenCL) host functions to access pinned buffers using standard pointers
#    fSourceA = (cl_float*)clEnqueueMapBuffer(cqCommandQueue[0], cmPinnedSrcA, CL_TRUE, CL_MAP_WRITE, 0, szBuffBytes, 0, NULL, NULL, &ciErrNum);
#    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

control_buf = cl.Buffer(cl_context,
                        cl.mem_flags.READ_ONLY | cl_mem_flags.HOST_WRITE_ONLY | cl_mem_flags.USE_HOST_PTR,
                        hostbuf=host_control_struct)

sparse_buf = cl.Buffer(cl_context,
                       cl.mem_flags.WRITE_ONLY | cl.mem_flags.HOST_READ_ONLY | cl.mem_flags.USE_HOST_PTR,
                       hostbuf=sparse_output)

#consider using fast relaxed math
#set block size equal to warp size
def build_file(cl_context, filename='cl_main.cl'):
    cl_file=open(filename, 'r')
    cl_program = cl.Program(cl_context, cl_file.read()).build()
    cl_file.close()
    return cl_program

cl_program = build_file(cl_context)


#change offset and shape to read portions of the mapped data
cl.enqueue_map_buffer(cl_queue, control_buf, cl.map_flags.WRITE,
                      offset=0, shape=host_control_struct.shape, dtype=host_control_struct.dtype, is_blocking=False)

#do stuff

cl.enqueue_map_buffer(cl_queue, control_buf, cl.map_flags.WRITE,
                      offset=0, shape=host_control_struct.shape, dtype=host_control_struct.dtype, is_blocking=False)

#for some reason, Nvidia seems to require 1 dimensional input
#cl_program.sum(cl_queue, a_np.shape, None, a_buf, b_buf, res_buf)
cl_program.sum(cl_queue, (a_np.size,), None, a_buf, b_buf, res_buf)


res_np = np.empty_like(a_np)
cl.enqueue_copy(cl_queue, res_np, res_buf)

import time
time.sleep(0.5)

for i in res_np:
    print(i)
