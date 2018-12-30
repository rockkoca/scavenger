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
) -> impl FnOnce() {
    move || {
        let mut gensig = [0u8; 32];
        let mut new_round = true;
        let mut last_buffer = None;
        let mut last_start_nonce = 0;
        let mut last_account_id = 0;
        for read_reply in rx_read_replies {
            let mut buffer = read_reply.buffer;
            if read_reply.len == 0 || benchmark {
                tx_empty_buffers.send(buffer);
                continue;
            }
          
            if *read_reply.gensig != gensig {
                gensig = *read_reply.gensig;
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
                new_round = false;
            } else {
                let result = gpu_transfer_and_hash(
                    context_mu.clone(),
                    buffer.get_gpu_buffers().unwrap(),
                    read_reply.len / 64,
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
                        reader_task_processed: false,
                        account_id: last_account_id,
                    })
                    .wait()
                    .expect("failed to send nonce data");

                last_buffer = buffer.get_gpu_data();
                last_start_nonce = read_reply.start_nonce;
                last_account_id = read_reply.account_id;
                new_round = false;
            }

            if read_reply.finished {
                    let result = gpu_hash(
                        context_mu.clone(),
                        read_reply.len / 64,
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
                        reader_task_processed: true,
                        account_id: last_account_id,
                    })
                    .wait()
                    .expect("failed to send nonce data");
            }

            tx_empty_buffers.send(buffer);
        }
    }
}
