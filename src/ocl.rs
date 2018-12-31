extern crate ocl_core as core;
use self::core::{
    ArgVal, ContextProperties, DeviceInfo, Event, KernelWorkGroupInfo, PlatformInfo, Status,
};

use config::Cfg;
use miner::Buffer;
use std::ffi::CString;
use std::process;
use std::slice::from_raw_parts_mut;
use std::sync::{Arc, Mutex};

static SRC: &'static str = include_str!("ocl/kernel.cl");
const SCOOP_SIZE: u64 = 64;

// convert the info or error to a string for printing:
macro_rules! to_string {
    ($expr:expr) => {
        match $expr {
            Ok(info) => info.to_string(),
            Err(err) => match err.api_status() {
                Some(Status::CL_KERNEL_ARG_INFO_NOT_AVAILABLE) => "Not available".into(),
                _ => err.to_string(),
            },
        }
    };
}

pub fn platform_info() {
    let platform_ids = core::get_platform_ids().unwrap();
    for (i, platform_id) in platform_ids.iter().enumerate() {
        info!(
            "OCL: platform {}, {} - {}",
            i,
            to_string!(core::get_platform_info(&platform_id, PlatformInfo::Name)),
            to_string!(core::get_platform_info(&platform_id, PlatformInfo::Version))
        );
        let device_ids = core::get_device_ids(&platform_id, None, None).unwrap();
        for (j, device_id) in device_ids.iter().enumerate() {
            info!(
                "OCL: device {}, {} - {}",
                j,
                to_string!(core::get_device_info(device_id, DeviceInfo::Vendor)),
                to_string!(core::get_device_info(device_id, DeviceInfo::Name))
            );
        }
    }
}

pub fn gpu_info(cfg: &Cfg) {
    if cfg.gpu_worker_thread_count > 0 {
        let platform_ids = core::get_platform_ids().unwrap();
        if cfg.gpu_platform >= platform_ids.len() {
            error!("OCL: Selected OpenCL platform doesn't exist. Shutting down...");
            process::exit(0);
        }
        let platform = platform_ids[cfg.gpu_platform];
        let device_ids = core::get_device_ids(&platform, None, None).unwrap();
        if cfg.gpu_device >= device_ids.len() {
            error!("OCL: Selected OpenCL device doesn't exist. Shutting down...");
            process::exit(0);
        }
        let device = device_ids[cfg.gpu_device];
        info!(
            "OCL: {} - {}",
            to_string!(core::get_platform_info(platform, PlatformInfo::Name)),
            to_string!(core::get_platform_info(platform, PlatformInfo::Version))
        );
        info!(
            "GPU: {} - {}",
            to_string!(core::get_device_info(&device, DeviceInfo::Vendor)),
            to_string!(core::get_device_info(&device, DeviceInfo::Name))
        );
        match core::get_device_info(&device, DeviceInfo::GlobalMemSize).unwrap() {
            core::DeviceInfoResult::GlobalMemSize(mem) => {
                info!(
                    "GPU: RAM={}MiB, Cores={}",
                    mem / 1024 / 1024,
                    to_string!(core::get_device_info(&device, DeviceInfo::MaxComputeUnits))
                );
                info!(
                    "GPU: RAM usage (estimated)={}MiB",
                    cfg.gpu_nonces_per_cache * 75 * 2 * cfg.gpu_worker_thread_count / 1024 / 1024
                        + cfg.gpu_worker_thread_count * 45
                );

                // yellow card
                if cfg.gpu_nonces_per_cache * 80 * 2 * cfg.gpu_worker_thread_count / 1024 / 1024
                    > mem as usize / 1024 / 1024
                {
                    warn!(
                        "GPU: Low on GPU memory. If your settings don't work, \
                         please reduce gpu_worker_threads and/or gpu_nonces_per_cache."
                    );
                }

                //red card
                if cfg.gpu_nonces_per_cache * 72 * 2 * cfg.gpu_worker_thread_count / 1024 / 1024
                    > mem as usize / 1024 / 1024
                {
                    error!(
                        "GPU: Insufficient GPU memory. Please reduce gpu_worker_threads \
                         and/or gpu_nonces_per_cache. Shutting down..."
                    );
                    process::exit(0);
                }
            }
            _ => panic!("Unexpected error. Can't obtain GPU memory size."),
        }
    } else if cfg.cpu_worker_thread_count == 0 {
        error!("CPU, GPU: no workers configured. Shutting down...");
        process::exit(0);
    }
}

pub struct GpuContext {
    context: core::Context,
    queue_a: core::CommandQueue,
    queue_b: core::CommandQueue,
    kernel1: core::Kernel,
    kernel2: core::Kernel,
    ldim1: [usize; 3],
    gdim1: [usize; 3],
    ldim2: [usize; 3],
    gdim2: [usize; 3],
    mapping: bool,
    gensig_gpu: core::Mem,
    deadlines_gpu: core::Mem,
    best_deadline_gpu: core::Mem,
    best_offset_gpu: core::Mem,
}

#[allow(dead_code)]
pub struct GpuBuffer {
    data: Arc<Mutex<Vec<u8>>>,
    buffer_ptr_host: Option<core::MemMap<u8>>,
    buffer_host: Option<core::Mem>,
    context: Arc<GpuContext>,
    data_gpu: core::Mem,
}

impl GpuContext {
    pub fn new(
        gpu_platform: usize,
        gpu_id: usize,
        nonces_per_cache: usize,
        mapping: bool,
    ) -> GpuContext {
        let platform_ids = core::get_platform_ids().unwrap();
        let platform_id = platform_ids[gpu_platform];
        let device_ids = core::get_device_ids(&platform_id, None, None).unwrap();
        let device_id = device_ids[gpu_id];
        let context_properties = ContextProperties::new().platform(platform_id);
        let context =
            core::create_context(Some(&context_properties), &[device_id], None, None).unwrap();
        let src_cstring = CString::new(SRC).unwrap();
        let program = core::create_program_with_source(&context, &[src_cstring]).unwrap();
        core::build_program(
            &program,
            None::<&[()]>,
            &CString::new("").unwrap(),
            None,
            None,
        )
        .unwrap();
        let queue_a = core::create_command_queue(&context, &device_id, None).unwrap();
        let queue_b = core::create_command_queue(&context, &device_id, None).unwrap();

        let kernel1 = core::create_kernel(&program, "calculate_deadlines").unwrap();
        let kernel2 = core::create_kernel(&program, "find_min").unwrap();

        let kernel1_workgroup_size = get_kernel_work_group_size(&kernel1, device_id);
        let kernel2_workgroup_size = get_kernel_work_group_size(&kernel2, device_id);
        info!("Kernel1 {}", kernel1_workgroup_size);
        info!("Kernel2 {}", kernel2_workgroup_size);
        let mut workgroup_count = nonces_per_cache / kernel1_workgroup_size;
        if nonces_per_cache % kernel1_workgroup_size != 0 {
            workgroup_count += 1;
        }

        let gdim1 = [kernel1_workgroup_size * workgroup_count, 1, 1];
        let ldim1 = [kernel1_workgroup_size, 1, 1];
        let gdim2 = [kernel2_workgroup_size, 1, 1];
        let ldim2 = [kernel2_workgroup_size, 1, 1];

        let gensig_gpu = unsafe {
            core::create_buffer::<_, u8>(&context, core::MEM_READ_ONLY, 32, None).unwrap()
        };

        let deadlines_gpu = unsafe {
            core::create_buffer::<_, u64>(&context, core::MEM_READ_WRITE, gdim1[0], None).unwrap()
        };

        let best_offset_gpu = unsafe {
            core::create_buffer::<_, u64>(&context, core::MEM_READ_WRITE, 1, None).unwrap()
        };

        let best_deadline_gpu = unsafe {
            core::create_buffer::<_, u64>(&context, core::MEM_READ_WRITE, 1, None).unwrap()
        };

        GpuContext {
            context,
            queue_a,
            queue_b,
            kernel1,
            kernel2,
            ldim1,
            gdim1,
            ldim2,
            gdim2,
            mapping,
            gensig_gpu,
            deadlines_gpu,
            best_deadline_gpu,
            best_offset_gpu,
        }
    }
}

impl GpuBuffer {
    pub fn new(context: &Arc<GpuContext>) -> Self {
        // todo
        let nvidia = true;
        // create buffers
        // mapping = zero copy buffers, no mapping = pinned memory for fast DMA.
        if context.mapping {
            let data_gpu = unsafe {
                core::create_buffer::<_, u8>(
                    &context.context,
                    core::MEM_READ_ONLY | core::MEM_ALLOC_HOST_PTR,
                    (SCOOP_SIZE as usize) * context.gdim1[0],
                    None,
                )
                .unwrap()
            };

            let mut buffer_ptr_host = unsafe {
                Some(
                    core::enqueue_map_buffer::<u8, _, _, _>(
                        &context.queue_b,
                        &data_gpu,
                        true,
                        core::MAP_WRITE,
                        0,
                        (SCOOP_SIZE as usize) * &context.gdim1[0],
                        None::<Event>,
                        None::<&mut Event>,
                    )
                    .unwrap(),
                )
            };

            let ptr = buffer_ptr_host.as_mut().unwrap().as_mut_ptr();
            let boxed_slice = unsafe {
                Box::<[u8]>::from_raw(from_raw_parts_mut(
                    ptr,
                    (SCOOP_SIZE as usize) * &context.gdim1[0],
                ))
            };
            let data = Arc::new(Mutex::new(boxed_slice.into_vec()));

            core::enqueue_unmap_mem_object(
                &context.queue_b,
                &data_gpu,
                buffer_ptr_host.as_ref().unwrap(),
                None::<Event>,
                None::<&mut Event>,
            )
            .unwrap();
            GpuBuffer {
                data,
                buffer_ptr_host: None,
                buffer_host: None,
                context: context.clone(),
                data_gpu,
            }
        } else {
            let buffer_host = unsafe {
                core::create_buffer::<_, u8>(
                    &context.context,
                    core::MEM_READ_ONLY | core::MEM_ALLOC_HOST_PTR,
                    (SCOOP_SIZE as usize) * context.gdim1[0],
                    None,
                )
                .unwrap()
            };
            let mut buffer_ptr_host = unsafe {
                Some(
                    core::enqueue_map_buffer::<u8, _, _, _>(
                        &context.queue_b,
                        &buffer_host,
                        true,
                        core::MAP_WRITE,
                        0,
                        (SCOOP_SIZE as usize) * context.gdim1[0],
                        None::<Event>,
                        None::<&mut Event>,
                    )
                    .unwrap(),
                )
            };
            let data_gpu = if nvidia {
                buffer_host.clone()
            } else {
                unsafe {
                    core::create_buffer::<_, u8>(
                        &context.context,
                        core::MEM_READ_ONLY,
                        (SCOOP_SIZE as usize) * context.gdim1[0],
                        None,
                    )
                    .unwrap()
                }
            };
            let buffer_host = if nvidia { None } else { Some(buffer_host) };

            let ptr = buffer_ptr_host.as_mut().unwrap().as_mut_ptr();
            let boxed_slice = unsafe {
                Box::<[u8]>::from_raw(from_raw_parts_mut(
                    ptr,
                    (SCOOP_SIZE as usize) * &context.gdim1[0],
                ))
            };
            let data = Arc::new(Mutex::new(boxed_slice.into_vec()));

            GpuBuffer {
                data,
                buffer_ptr_host,
                buffer_host,
                context: context.clone(),
                data_gpu,
            }
        }
    }
}

impl Buffer for GpuBuffer {
    fn get_buffer_for_writing(&mut self) -> Arc<Mutex<Vec<u8>>> {
        if self.context.mapping {
            unsafe {
                self.buffer_ptr_host = Some(
                    core::enqueue_map_buffer::<u8, _, _, _>(
                        &self.context.queue_b,
                        &self.data_gpu,
                        true,
                        core::MAP_WRITE,
                        0,
                        (SCOOP_SIZE as usize) * self.context.gdim1[0],
                        None::<Event>,
                        None::<&mut Event>,
                    ).unwrap(),
                );
            }
        }
        self.data.clone()
    }

    fn get_buffer(&mut self) -> Arc<Mutex<Vec<u8>>> {
        self.data.clone()
    }

    fn get_gpu_context(&self) -> Option<Arc<GpuContext>> {
        Some(self.context.clone())
    }
    fn get_gpu_buffers(&self) -> Option<&GpuBuffer> {
        Some(self)
    }
    fn get_gpu_data(&self) -> Option<core::Mem> {
        Some(self.data_gpu.clone())
    }
}

// Ohne Gummi im Bahnhofsviertel... das wird noch Konsequenzen haben
unsafe impl Sync for GpuContext {}
unsafe impl Send for GpuBuffer {}

pub fn gpu_transfer(gpu_context: Arc<GpuContext>, buffer: &GpuBuffer, gensig: [u8; 32]) {
    unsafe {
        core::enqueue_write_buffer(
            &gpu_context.queue_a,
            &gpu_context.gensig_gpu,
            true,
            0,
            &gensig,
            None::<Event>,
            None::<&mut Event>,
        )
        .unwrap();
    }
    let data = buffer.data.clone();
    let data2 = (*data).lock().unwrap();
    if gpu_context.mapping {
        let temp2 = buffer.buffer_ptr_host.as_ref().unwrap();
        core::enqueue_unmap_mem_object(
            &gpu_context.queue_b,
            &buffer.data_gpu,
            &*temp2,
            None::<Event>,
            None::<&mut Event>,
        )
        .unwrap();
    } else {
        unsafe {
            core::enqueue_write_buffer(
                &gpu_context.queue_b,
                &buffer.data_gpu,
                true,
                0,
                &data2,
                None::<Event>,
                None::<&mut Event>,
            )
            .unwrap();
        }
    }
}

pub fn gpu_transfer_and_hash(
    gpu_context: Arc<GpuContext>,
    buffer: &GpuBuffer,
    nonce_count: usize,
    data_gpu: &core::Mem,
) -> (u64, u64) {
    let data = buffer.data.clone();
    let data2 = (*data).lock().unwrap();

    // transfer
    if gpu_context.mapping {
           let temp2 = buffer.buffer_ptr_host.as_ref().unwrap();
        core::enqueue_unmap_mem_object(
            &gpu_context.queue_b,
            &buffer.data_gpu,
            &*temp2,
            None::<Event>,
            None::<&mut Event>,
        )
        .unwrap();
    } else {
        unsafe {
            core::enqueue_write_buffer(
                &gpu_context.queue_b,
                &buffer.data_gpu,
                false,
                0,
                &data2,
                None::<Event>,
                None::<&mut Event>,
            )
            .unwrap();
        }
    }

    core::set_kernel_arg(
        &gpu_context.kernel1,
        0,
        ArgVal::mem(&gpu_context.gensig_gpu),
    )
    .unwrap();
    core::set_kernel_arg(&gpu_context.kernel1, 1, ArgVal::mem(&data_gpu)).unwrap();
    core::set_kernel_arg(
        &gpu_context.kernel1,
        2,
        ArgVal::mem(&gpu_context.deadlines_gpu),
    )
    .unwrap();

    unsafe {
        core::enqueue_kernel(
            &gpu_context.queue_a,
            &gpu_context.kernel1,
            1,
            None,
            &gpu_context.gdim1,
            Some(gpu_context.ldim1),
            None::<Event>,
            None::<&mut Event>,
        )
        .unwrap();
    }

    core::set_kernel_arg(
        &gpu_context.kernel2,
        0,
        ArgVal::mem(&gpu_context.deadlines_gpu),
    )
    .unwrap();
    core::set_kernel_arg(
        &gpu_context.kernel2,
        1,
        ArgVal::primitive(&(nonce_count as u64)),
    )
    .unwrap();
    core::set_kernel_arg(
        &gpu_context.kernel2,
        2,
        ArgVal::local::<u32>(&gpu_context.ldim2[0]),
    )
    .unwrap();
    core::set_kernel_arg(
        &gpu_context.kernel2,
        3,
        ArgVal::mem(&gpu_context.best_offset_gpu),
    )
    .unwrap();
    core::set_kernel_arg(
        &gpu_context.kernel2,
        4,
        ArgVal::mem(&gpu_context.best_deadline_gpu),
    )
    .unwrap();

    unsafe {
        core::enqueue_kernel(
            &gpu_context.queue_a,
            &gpu_context.kernel2,
            1,
            None,
            &gpu_context.gdim2,
            Some(gpu_context.ldim2),
            None::<Event>,
            None::<&mut Event>,
        )
        .unwrap();
    }

    let mut best_offset = vec![0u64; 1];
    let mut best_deadline = vec![0u64; 1];

    unsafe {
        core::enqueue_read_buffer(
            &gpu_context.queue_a,
            &gpu_context.best_offset_gpu,
            true,
            0,
            &mut best_offset,
            None::<Event>,
            None::<&mut Event>,
        )
        .unwrap();
    }
    unsafe {
        core::enqueue_read_buffer(
            &gpu_context.queue_a,
            &gpu_context.best_deadline_gpu,
            true,
            0,
            &mut best_deadline,
            None::<Event>,
            None::<&mut Event>,
        )
        .unwrap();
    }
    // wait for transfer
    core::finish(&gpu_context.queue_b).unwrap();

    (best_deadline[0], best_offset[0])
}

pub fn gpu_hash(
    gpu_context: Arc<GpuContext>,
    nonce_count: usize,
    data_gpu: &core::Mem,
) -> (u64, u64) {
    core::set_kernel_arg(
        &gpu_context.kernel1,
        0,
        ArgVal::mem(&gpu_context.gensig_gpu),
    )
    .unwrap();
    core::set_kernel_arg(&gpu_context.kernel1, 1, ArgVal::mem(&data_gpu)).unwrap();
    core::set_kernel_arg(
        &gpu_context.kernel1,
        2,
        ArgVal::mem(&gpu_context.deadlines_gpu),
    )
    .unwrap();

    unsafe {
        core::enqueue_kernel(
            &gpu_context.queue_a,
            &gpu_context.kernel1,
            1,
            None,
            &gpu_context.gdim1,
            Some(gpu_context.ldim1),
            None::<Event>,
            None::<&mut Event>,
        )
        .unwrap();
    }

    core::set_kernel_arg(
        &gpu_context.kernel2,
        0,
        ArgVal::mem(&gpu_context.deadlines_gpu),
    )
    .unwrap();
    core::set_kernel_arg(
        &gpu_context.kernel2,
        1,
        ArgVal::primitive(&(nonce_count as u64)),
    )
    .unwrap();
    core::set_kernel_arg(
        &gpu_context.kernel2,
        2,
        ArgVal::local::<u32>(&gpu_context.ldim2[0]),
    )
    .unwrap();
    core::set_kernel_arg(
        &gpu_context.kernel2,
        3,
        ArgVal::mem(&gpu_context.best_offset_gpu),
    )
    .unwrap();
    core::set_kernel_arg(
        &gpu_context.kernel2,
        4,
        ArgVal::mem(&gpu_context.best_deadline_gpu),
    )
    .unwrap();

    unsafe {
        core::enqueue_kernel(
            &gpu_context.queue_a,
            &gpu_context.kernel2,
            1,
            None,
            &gpu_context.gdim2,
            Some(gpu_context.ldim2),
            None::<Event>,
            None::<&mut Event>,
        )
        .unwrap();
    }

    let mut best_offset = vec![0u64; 1];
    let mut best_deadline = vec![0u64; 1];

    unsafe {
        core::enqueue_read_buffer(
            &gpu_context.queue_a,
            &gpu_context.best_offset_gpu,
            true,
            0,
            &mut best_offset,
            None::<Event>,
            None::<&mut Event>,
        )
        .unwrap();
    }
    unsafe {
        core::enqueue_read_buffer(
            &gpu_context.queue_a,
            &gpu_context.best_deadline_gpu,
            true,
            0,
            &mut best_deadline,
            None::<Event>,
            None::<&mut Event>,
        )
        .unwrap();
    }

    (best_deadline[0], best_offset[0])
}

fn get_kernel_work_group_size(x: &core::Kernel, y: core::DeviceId) -> usize {
    match core::get_kernel_work_group_info(x, y, KernelWorkGroupInfo::WorkGroupSize).unwrap() {
        core::KernelWorkGroupInfoResult::WorkGroupSize(kws) => kws,
        _ => panic!("Unexpected error"),
    }
}
