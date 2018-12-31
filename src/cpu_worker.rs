use chan;
use futures::sync::mpsc;
use futures::{Future, Sink};
use libc::{c_void, uint64_t};
use miner::{Buffer, NonceData};

use reader::ReadReply;
use std::u64;

extern "C" {
    pub fn find_best_deadline_sph(
        scoops: *mut c_void,
        nonce_count: uint64_t,
        gensig: *const c_void,
        best_deadline: *mut uint64_t,
        best_offset: *mut uint64_t,
    ) -> ();
}

cfg_if! {
    if #[cfg(feature = "simd")] {
        extern "C" {
            pub fn find_best_deadline_avx512f(
                scoops: *mut c_void,
                nonce_count: uint64_t,
                gensig: *const c_void,
                best_deadline: *mut uint64_t,
                best_offset: *mut uint64_t,
            ) -> ();

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
    }
}

cfg_if! {
    if #[cfg(feature = "neon")] {
        extern "C" {
            pub fn find_best_deadline_neon(
                scoops: *mut c_void,
                nonce_count: uint64_t,
                gensig: *const c_void,
                best_deadline: *mut uint64_t,
                best_offset: *mut uint64_t,
            ) -> ();
        }
    }
}

pub fn create_cpu_worker_task(
    benchmark: bool,
    thread_pool: rayon::ThreadPool,
    rx_read_replies: chan::Receiver<ReadReply>,
    tx_empty_buffers: chan::Sender<Box<Buffer + Send>>,
    tx_nonce_data: mpsc::Sender<NonceData>,
) -> impl FnOnce() {
    move || {
        for read_reply in rx_read_replies {
            let task = hash(
                read_reply,
                tx_empty_buffers.clone(),
                tx_nonce_data.clone(),
                benchmark,
            );

            thread_pool.spawn(task);
        }
    }
}

pub fn hash(
    read_reply: ReadReply,
    tx_empty_buffers: chan::Sender<Box<Buffer + Send>>,
    tx_nonce_data: mpsc::Sender<NonceData>,
    benchmark: bool,
) -> impl FnOnce() {
    move || {
        let mut buffer = read_reply.buffer;
        if read_reply.info.len == 0 || benchmark {
            tx_empty_buffers.send(buffer).unwrap();
            return;
        }

        let mut deadline: u64 = u64::MAX;
        let mut offset: u64 = 0;

        let mut_bs = buffer.get_buffer_for_writing();
        let mut bs = mut_bs.lock().unwrap();
        // todo wrong place for padding. reader should take care of padding, we shouldn't need mut here
        let padded = pad(&mut bs, read_reply.info.len, 8 * 64);
        #[cfg(feature = "simd")]
        unsafe {
            if is_x86_feature_detected!("avx512f") {
                find_best_deadline_avx512f(
                    bs.as_ptr() as *mut c_void,
                    (read_reply.info.len as u64 + padded as u64) / 64,
                    read_reply.info.gensig.as_ptr() as *const c_void,
                    &mut deadline,
                    &mut offset,
                );
            } else if is_x86_feature_detected!("avx2") {
                find_best_deadline_avx2(
                    bs.as_ptr() as *mut c_void,
                    (read_reply.info.len as u64 + padded as u64) / 64,
                    read_reply.info.gensig.as_ptr() as *const c_void,
                    &mut deadline,
                    &mut offset,
                );
            } else if is_x86_feature_detected!("avx") {
                find_best_deadline_avx(
                    bs.as_ptr() as *mut c_void,
                    (read_reply.info.len as u64 + padded as u64) / 64,
                    read_reply.info.gensig.as_ptr() as *const c_void,
                    &mut deadline,
                    &mut offset,
                );
            } else if is_x86_feature_detected!("sse2") {
                find_best_deadline_sse2(
                    bs.as_ptr() as *mut c_void,
                    (read_reply.info.len as u64 + padded as u64) / 64,
                    read_reply.info.gensig.as_ptr() as *const c_void,
                    &mut deadline,
                    &mut offset,
                );
            } else {
                find_best_deadline_sph(
                    bs.as_ptr() as *mut c_void,
                    (read_reply.info.len as u64 + padded as u64) / 64,
                    read_reply.info.gensig.as_ptr() as *const c_void,
                    &mut deadline,
                    &mut offset,
                );
            }
        }
        #[cfg(feature = "neon")]
        unsafe {
            #[cfg(target_arch = "arm")]
            let neon = is_arm_feature_detected!("neon");
            #[cfg(target_arch = "aarch64")]
            let neon = true;
            if neon {
                find_best_deadline_neon(
                    bs.as_ptr() as *mut c_void,
                    (read_reply.len as u64 + padded as u64) / 64,
                    read_reply.gensig.as_ptr() as *const c_void,
                    &mut deadline,
                    &mut offset,
                );
            } else {
                find_best_deadline_sph(
                    bs.as_ptr() as *mut c_void,
                    (read_reply.len as u64 + padded as u64) / 64,
                    read_reply.gensig.as_ptr() as *const c_void,
                    &mut deadline,
                    &mut offset,
                );
            }
        }
        #[cfg(not(any(feature = "simd", feature = "neon")))]
        unsafe {
            find_best_deadline_sph(
                bs.as_ptr() as *mut c_void,
                (read_reply.len as u64 + padded as u64) / 64,
                read_reply.gensig.as_ptr() as *const c_void,
                &mut deadline,
                &mut offset,
            );
        }

        tx_nonce_data
            .clone()
            .send(NonceData {
                height: read_reply.info.height,
                deadline,
                nonce: offset + read_reply.info.start_nonce,
                reader_task_processed: read_reply.info.finished,
                account_id: read_reply.info.account_id,
            })
            .wait()
            .expect("failed to send nonce data");
        tx_empty_buffers.send(buffer).unwrap();
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
