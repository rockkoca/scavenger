use chan;
use futures::sync::mpsc;
use futures::{Future, Sink};
use miner::{Buffer, NonceData};
use ocl::GpuContext;
use ocl::{gpu_hash, gpu_transfer, gpu_transfer_and_hash};
use reader::ReadReply;
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
        let mut gensig = [0u8; 32];
        let mut new_round = true;
        let mut last_buffer = None;
        let mut last_start_nonce = 0;
        let mut last_account_id = 0;
        let mut last_nonces = 0;
        let mut last_height = 0;
        let mut last_finished = false;
        let mut drive_count = 0;
        for read_reply in rx_read_replies {
            let mut buffer = read_reply.buffer;
            if read_reply.len == 0 || benchmark {
                if read_reply.height == 1 {
                    drive_count = 0;
                    new_round = true;
                    //   info!("START XCount{}", drive_count);
                }
                if read_reply.height == 0 {
                    drive_count += 1;
                    if drive_count == num_drives {
                        //  info!("FINAL XCount{}", drive_count);
                        if !new_round {
                            let result = gpu_hash(
                                context_mu.clone(),
                                last_nonces,
                                last_buffer.as_ref().unwrap(),
                            );
                            let deadline = result.0;
                            let offset = result.1;

                            tx_nonce_data
                                .clone()
                                .send(NonceData {
                                    height: last_height,
                                    deadline,
                                    nonce: offset + last_start_nonce,
                                    reader_task_processed: last_finished,
                                    account_id: last_account_id,
                                })
                                .wait()
                                .expect("failed to send nonce data");
                        }
                    }
                }

                tx_empty_buffers.send(buffer).unwrap();
                continue;
            }

            if *read_reply.gensig != gensig {
                gensig = *read_reply.gensig;
                last_height = read_reply.height;
                new_round = true;
            }

            if new_round {
                gpu_transfer(
                    context_mu.clone(),
                    buffer.get_gpu_buffers().unwrap(),
                    *read_reply.gensig,
                );
                last_buffer = buffer.get_gpu_data();
                last_start_nonce = read_reply.start_nonce;
                last_account_id = read_reply.account_id;
                last_nonces = read_reply.len / 64;
                last_finished = read_reply.finished;
                new_round = false;
            } else {
                let result = gpu_transfer_and_hash(
                    context_mu.clone(),
                    buffer.get_gpu_buffers().unwrap(),
                    last_nonces,
                    last_buffer.as_ref().unwrap(),
                );
                let deadline = result.0;
                let offset = result.1;

                tx_nonce_data
                    .clone()
                    .send(NonceData {
                        height: read_reply.height,
                        deadline,
                        nonce: offset + last_start_nonce,
                        reader_task_processed: last_finished,
                        account_id: last_account_id,
                    })
                    .wait()
                    .expect("failed to send nonce data");

                last_buffer = buffer.get_gpu_data();
                last_start_nonce = read_reply.start_nonce;
                last_account_id = read_reply.account_id;
                last_nonces = read_reply.len / 64;
                last_finished = read_reply.finished;
                new_round = false;
            }

            tx_empty_buffers.send(buffer).unwrap();
        }
    }
}
