use chan;
use futures::sync::mpsc;
use futures::{Future, Sink};
use libc::{c_void, uint64_t};
use ocl;

use reader::{Buffer, ReadInfo, WritableBuffer};
use std::u64;

extern "C" {
    pub fn find_best_deadline_avx2(
        scoops: *mut c_void,
        nonce_count: uint64_t,
        gensig: *const c_void,
        best_deadline: *mut uint64_t,
        best_offset: *mut uint64_t,
    ) -> ();

    pub fn find_best_deadline_avx(
        scoops: *mut c_void,
        nonce_count: uint64_t,
        gensig: *const c_void,
        best_deadline: *mut uint64_t,
        best_offset: *mut uint64_t,
    ) -> ();

    pub fn find_best_deadline_sse2(
        scoops: *mut c_void,
        nonce_count: uint64_t,
        gensig: *const c_void,
        best_deadline: *mut uint64_t,
        best_offset: *mut uint64_t,
    ) -> ();
}

pub struct NonceData {
    pub height: u64,
    pub deadline: u64,
    pub nonce: u64,
    pub reader_task_processed: bool,
}

pub fn create_worker_task(
    rx_read_replies: chan::Receiver<(Box<Buffer>, ReadInfo)>,
    tx_empty_buffers: chan::Sender<Box<WritableBuffer + Send>>,
    tx_nonce_data: mpsc::Sender<NonceData>,
) -> impl FnOnce() {
    let find_best_deadline = if is_x86_feature_detected!("avx2") {
        find_best_deadline_avx2
    } else if is_x86_feature_detected!("avx") {
        find_best_deadline_avx
    } else {
        find_best_deadline_sse2
    };

    move || {
        for (mut buffer, read_info) in rx_read_replies {
            if read_info.len == 0 {
                match *buffer {
                    Buffer::Cpu(buffer) => {
                        tx_empty_buffers.send(Box::new(buffer) as Box<WritableBuffer + Send>)
                    }
                    Buffer::Gpu(buffer) => {
                        tx_empty_buffers.send(Box::new(buffer) as Box<WritableBuffer + Send>)
                    }
                }
                continue;
            }

            let mut deadline: u64 = u64::MAX;
            let mut offset: u64 = 0;
            let buffer = match *buffer {
                Buffer::Cpu(mut buffer) => {
                    let mut_bs = buffer.get_buffer();
                    let mut bs = mut_bs.lock().unwrap();
                    let padded = pad(&mut bs, read_info.len, 8 * 64);
                    unsafe {
                        find_best_deadline(
                            bs.as_ptr() as *mut c_void,
                            (read_info.len as u64 + padded as u64) / 64,
                            read_info.gensig.as_ptr() as *const c_void,
                            &mut deadline,
                            &mut offset,
                        )
                    }

                    Box::new(buffer) as Box<WritableBuffer + Send>
                }
                Buffer::Gpu(buffer) => {
                    let deadline_and_offset =
                        ocl::find_best_deadline_gpu(&buffer, read_info.len / 64, *read_info.gensig);
                    deadline = deadline_and_offset.0;
                    offset = deadline_and_offset.1;

                    Box::new(buffer) as Box<WritableBuffer + Send>
                }
            };

            tx_nonce_data
                .clone()
                .send(NonceData {
                    height: read_info.height,
                    deadline,
                    nonce: offset + read_info.start_nonce,
                    reader_task_processed: read_info.finished,
                }).wait()
                .expect("failed to send nonce data");
            tx_empty_buffers.send(buffer);
        }
    }
}

pub fn pad(b: &mut [u8], l: usize, p: usize) -> usize {
    let r = p - l % p;
    if r != p {
        for i in 0..r {
            b[i] = b[0];
        }
        r
    } else {
        0
    }
}
