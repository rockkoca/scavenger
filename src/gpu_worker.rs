use chan;
use futures::sync::mpsc;
use futures::{Future, Sink};
use miner::{Buffer, NonceData};
use ocl;
use reader::ReadReply;
use std::u64;
use std::sync::{Arc,Mutex};
use ocl::GpuContext;


pub fn create_gpu_worker_task(
    benchmark: bool,
    rx_read_replies: chan::Receiver<ReadReply>,
    tx_empty_buffers: chan::Sender<Box<Buffer + Send>>,
    tx_nonce_data: mpsc::Sender<NonceData>,
    context_mu: Arc<Mutex<GpuContext>>
) -> impl FnOnce() {
    move || {
        for read_reply in rx_read_replies {
            let mut buffer = read_reply.buffer;
            if read_reply.len == 0 {
                tx_empty_buffers.send(buffer);
                continue;
            }

            let mut deadline: u64 = u64::MAX;
            let mut offset: u64 = 0;

            if !benchmark {
                let tuple = ocl::find_best_deadline_gpu(
                    context_mu.clone(),
                    buffer.get_gpu_buffers().unwrap(),
                    read_reply.len / 64,
                    *read_reply.gensig,
                );
                deadline = tuple.0;
                offset = tuple.1;
            }

            tx_nonce_data
                .clone()
                .send(NonceData {
                    height: read_reply.height,
                    deadline,
                    nonce: offset + read_reply.start_nonce,
                    reader_task_processed: read_reply.finished,
                    account_id: read_reply.account_id,
                }).wait()
                .expect("failed to send nonce data");
            tx_empty_buffers.send(buffer);
        }
    }
}