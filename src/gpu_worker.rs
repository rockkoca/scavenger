use chan;
use futures::sync::mpsc;
use futures::{Future, Sink};
use miner::{Buffer, NonceData};
use ocl::GpuContext;
use ocl::{gpu_hash, gpu_transfer, gpu_transfer_and_hash};
use reader::{BufferInfo, ReadReply};
use std::sync::Arc;

pub fn create_gpu_worker_task(
    benchmark: bool,
    rx_read_replies: chan::Receiver<ReadReply>,
    tx_empty_buffers: chan::Sender<Box<Buffer + Send>>,
    tx_nonce_data: mpsc::Sender<NonceData>,
    context_mu: Arc<GpuContext>,
    num_drives: usize,
) -> impl FnOnce() {
    move || {
        let mut new_round = true;
        let mut last_buffer_a = None;
        let mut last_buffer_info_a = BufferInfo {
            len: 0,
            height: 0,
            gensig: Arc::new([0u8; 32]),
            start_nonce: 0,
            finished: false,
            account_id: 0,
        };
        let mut drive_count = 0;

        for read_reply in rx_read_replies {
            let mut buffer = read_reply.buffer;

            if read_reply.info.len == 0 || benchmark {
                if read_reply.info.height == 1 {
                    drive_count = 0;
                    new_round = true;
                }
                if read_reply.info.height == 0 {
                    drive_count += 1;
                    if drive_count == num_drives {
                        if !new_round {
                            let result = gpu_hash(
                                context_mu.clone(),
                                last_buffer_info_a.len / 64,
                                last_buffer_a.as_ref().unwrap(),
                                true,
                            );
                            let deadline = result.0;
                            let offset = result.1;

                            tx_nonce_data
                                .clone()
                                .send(NonceData {
                                    height: last_buffer_info_a.height,
                                    deadline,
                                    nonce: offset + last_buffer_info_a.start_nonce,
                                    reader_task_processed: last_buffer_info_a.finished,
                                    account_id: last_buffer_info_a.account_id,
                                })
                                .wait()
                                .expect("failed to send nonce data");
                        }
                    }
                }

                tx_empty_buffers.send(buffer).unwrap();
                continue;
            }

            if new_round {
                gpu_transfer(
                    context_mu.clone(),
                    buffer.get_gpu_buffers().unwrap(),
                    *read_reply.info.gensig,
                );
            } else {
                let result = gpu_transfer_and_hash(
                    context_mu.clone(),
                    buffer.get_gpu_buffers().unwrap(),
                    last_buffer_info_a.len / 64,
                    last_buffer_a.as_ref().unwrap(),
                );
                let deadline = result.0;
                let offset = result.1;

                tx_nonce_data
                    .clone()
                    .send(NonceData {
                        height: last_buffer_info_a.height,
                        deadline,
                        nonce: offset + last_buffer_info_a.start_nonce,
                        reader_task_processed: last_buffer_info_a.finished,
                        account_id: last_buffer_info_a.account_id,
                    })
                    .wait()
                    .expect("failed to send nonce data");
            }
            last_buffer_a = buffer.get_gpu_data();
            last_buffer_info_a = read_reply.info;
            new_round = false;
            tx_empty_buffers.send(buffer).unwrap();
        }
    }
}
