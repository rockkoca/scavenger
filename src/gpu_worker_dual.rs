use chan;
use futures::sync::mpsc;
use futures::{Future, Sink};
use miner::{Buffer, NonceData};
use ocl::GpuContext;
use ocl::{gpu_hash, gpu_hash_dual, gpu_transfer, gpu_transfer_and_hash_dual, gpu_transfer_dual};
use reader::{BufferInfo, ReadReply};
use std::sync::Arc;

pub fn create_gpu_worker_dual_task(
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
        let mut last_buffer_b = None;
        let mut last_buffer_info_a = BufferInfo {
            len: 0,
            height: 0,
            gensig: Arc::new([0u8; 32]),
            start_nonce: 0,
            finished: false,
            account_id: 0,
        };
        let mut last_buffer_info_b = BufferInfo {
            len: 0,
            height: 0,
            gensig: Arc::new([0u8; 32]),
            start_nonce: 0,
            finished: false,
            account_id: 0,
        };
        let mut drive_count = 0;
        // todo remove
        info!("DUAL");
        for read_reply_a in &rx_read_replies {
            let mut buffer_a = read_reply_a.buffer;

            // check signals
            if read_reply_a.info.len == 0 || benchmark {
                // start signal
                if read_reply_a.info.height == 1 {
                    drive_count = 0;
                    new_round = true;
                }
                // termination signal: increase drive counter. if all drives done hash last buffer and end.
                if read_reply_a.info.height == 0 {
                    drive_count += 1;
                    if drive_count == num_drives {
                        if !new_round {
                            let result = gpu_hash_dual(
                                context_mu.clone(),
                                last_buffer_info_a.len / 64,
                                last_buffer_a.as_ref().unwrap(),
                                last_buffer_info_a.len / 64,
                                last_buffer_a.as_ref().unwrap(),
                            );
                            let deadline_a = result.0;
                            let offset_a = result.1;
                            let deadline_b = result.2;
                            let offset_b = result.3;
                            tx_nonce_data
                                .clone()
                                .send(NonceData {
                                    height: last_buffer_info_a.height,
                                    deadline: deadline_a,
                                    nonce: offset_a + last_buffer_info_a.start_nonce,
                                    reader_task_processed: last_buffer_info_a.finished,
                                    account_id: last_buffer_info_a.account_id,
                                })
                                .wait()
                                .expect("failed to send nonce data");
                            tx_nonce_data
                                .clone()
                                .send(NonceData {
                                    height: last_buffer_info_b.height,
                                    deadline: deadline_b,
                                    nonce: offset_b + last_buffer_info_b.start_nonce,
                                    reader_task_processed: last_buffer_info_b.finished,
                                    account_id: last_buffer_info_b.account_id,
                                })
                                .wait()
                                .expect("failed to send nonce data");
                        }
                    }
                }
                tx_empty_buffers.send(buffer_a).unwrap();
                continue;
            }

            // dual-copy mode: get second read_reply
            let read_reply_b = rx_read_replies.recv().unwrap();
            let mut buffer_b = read_reply_b.buffer;

            // check signals
            if read_reply_b.info.len == 0 {
                // start signal
                if read_reply_b.info.height == 1 {
                    drive_count = 0;
                    new_round = true;
                }
                // termination signal: increase drive counter. if all drives done transfer & hash last buffer and end. has previous two as well
                if read_reply_b.info.height == 0 {
                    drive_count += 1;
                    if drive_count == num_drives {
                        if !new_round {
                            // hash last buffer
                            let result = gpu_hash_dual(
                                context_mu.clone(),
                                last_buffer_info_a.len / 64,
                                last_buffer_a.as_ref().unwrap(),
                                last_buffer_info_b.len / 64,
                                last_buffer_b.as_ref().unwrap(),
                            );
                            let deadline_a = result.0;
                            let offset_a = result.1;
                            let deadline_b = result.2;
                            let offset_b = result.3;
                            tx_nonce_data
                                .clone()
                                .send(NonceData {
                                    height: last_buffer_info_a.height,
                                    deadline: deadline_a,
                                    nonce: offset_a + last_buffer_info_a.start_nonce,
                                    reader_task_processed: last_buffer_info_a.finished,
                                    account_id: last_buffer_info_a.account_id,
                                })
                                .wait()
                                .expect("failed to send nonce data");
                            tx_nonce_data
                                .clone()
                                .send(NonceData {
                                    height: last_buffer_info_b.height,
                                    deadline: deadline_b,
                                    nonce: offset_b + last_buffer_info_b.start_nonce,
                                    reader_task_processed: last_buffer_info_b.finished,
                                    account_id: last_buffer_info_b.account_id,
                                })
                                .wait()
                                .expect("failed to send nonce data");

                            // transfer and hash buffer_a
                            gpu_transfer(
                                context_mu.clone(),
                                buffer_a.get_gpu_buffers().unwrap(),
                                *read_reply_a.info.gensig,
                            );
                            let result = gpu_hash(
                                context_mu.clone(),
                                read_reply_a.info.len / 64,
                                buffer_a.get_gpu_data().as_ref().unwrap(),
                            );
                            let deadline = result.0;
                            let offset = result.1;

                            tx_nonce_data
                                .clone()
                                .send(NonceData {
                                    height: read_reply_a.info.height,
                                    deadline,
                                    nonce: offset + read_reply_a.info.start_nonce,
                                    reader_task_processed: read_reply_a.info.finished,
                                    account_id: read_reply_a.info.account_id,
                                })
                                .wait()
                                .expect("failed to send nonce data");
                        }
                    }
                }
                tx_empty_buffers.send(buffer_a).unwrap();
                tx_empty_buffers.send(buffer_b).unwrap();
                continue;
            }

            if new_round {
                gpu_transfer_dual(
                    context_mu.clone(),
                    buffer_a.get_gpu_buffers().unwrap(),
                    buffer_b.get_gpu_buffers().unwrap(),
                    *read_reply_a.info.gensig,
                );
                last_buffer_a = buffer_a.get_gpu_data();
                last_buffer_info_a = read_reply_a.info;

                last_buffer_b = buffer_b.get_gpu_data();
                last_buffer_info_b = read_reply_b.info;

                new_round = false;
            } else {
                let result = gpu_transfer_and_hash_dual(
                    context_mu.clone(),
                    buffer_a.get_gpu_buffers().unwrap(),
                    last_buffer_info_a.len / 64,
                    last_buffer_a.as_ref().unwrap(),
                    buffer_b.get_gpu_buffers().unwrap(),
                    last_buffer_info_b.len / 64,
                    last_buffer_b.as_ref().unwrap(),
                );
                let deadline_a = result.0;
                let offset_a = result.1;
                let deadline_b = result.2;
                let offset_b = result.3;
                tx_nonce_data
                    .clone()
                    .send(NonceData {
                        height: last_buffer_info_a.height,
                        deadline: deadline_a,
                        nonce: offset_a + last_buffer_info_a.start_nonce,
                        reader_task_processed: last_buffer_info_a.finished,
                        account_id: last_buffer_info_a.account_id,
                    })
                    .wait()
                    .expect("failed to send nonce data");
                tx_nonce_data
                    .clone()
                    .send(NonceData {
                        height: last_buffer_info_b.height,
                        deadline: deadline_b,
                        nonce: offset_b + last_buffer_info_b.start_nonce,
                        reader_task_processed: last_buffer_info_b.finished,
                        account_id: last_buffer_info_b.account_id,
                    })
                    .wait()
                    .expect("failed to send nonce data");
                last_buffer_a = buffer_a.get_gpu_data();
                last_buffer_info_a = read_reply_a.info;
                last_buffer_b = buffer_b.get_gpu_data();
                last_buffer_info_b = read_reply_b.info;
                new_round = false;
            }
            tx_empty_buffers.send(buffer_a).unwrap();
            tx_empty_buffers.send(buffer_b).unwrap();
        }
    }
}
