typedef enum tc_codes{
    HOST_REQUESTED_END = 1<<0 //host requested end of operations, so now we can clean up.
} thread_control_codes;

typedef struct pt_control{
    int time_before_restart; //time in milliseconds before the thread is shut down to avoid watchdog timeout
    thread_control_codes codes;

} persistent_thread_control;

__kernel void no_svm_async_write(__global int* data_in){
    event_t async_work_group_copy
}

//Note: Because these are persistent, only a certain amount can run at a time, so don't try to run 10000 kernels when
// your GPU only supports 900 at a time. Also, try to specify time_before_restart as -1 if there's no GPU timeout,
// and just under the GPU timeout if there is one. There is no Opencl clock, but Nvidia Opencl has one, there's
// clGetEventProfilingInfoand I can make the code guess the time and stay under the limit a specific percentage
// of the time.
__kernel void persistent_htm_1d(
    __global persistent_thread_control* cl_control,
    __global int* output
    )
{
    int gid;
    global_id = get_global_id(0);
    //since each group executes the same instructions, this is needed to know when to execute different instructions.
    //a warp would be better, but Opencl doesn't have that. Todo: NVidia OpenCL might have warp equivalents.
    group_id = get_group_id(0);
    //this is needed for dividing a problem into equally sized chunks.
    num_groups = get_num_groups(0);
    //todo: reduce global access by reducing the number of times we check cl_control here,
    //todo:  and make sure it's a broadcast
    while(!(cl_control->codes&thread_control_codes&HOST_REQUESTED_END)){

    }
    //res_buf[gid] = distance(a_buf[gid],b_buf[gid]);
    res_buf[gid] = get_group_id(0);//since SIMT executes the same instructions on each work group, synchronize based on work group id
    //subgroups would be preferred, but Nvidia doesn't implement this

}