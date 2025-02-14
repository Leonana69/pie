use crate::state::{
    BlockError, CausalLanguageModel, CausalTransformer, ImageEmbedder, KvBlock, ObjectAllocator,
    ObjectId, StreamId, TokenDist, TokenEmb,
};
use crate::utils::IdPool;
use futures::TryFutureExt;
use prost::Message;
use std::collections::HashMap;
use std::mem;
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, ZmqError, ZmqMessage};

mod sdi {
    include!(concat!(env!("OUT_DIR"), "/sdi.rs"));
}

/// Intermediate representation of a command to be executed by the backend.
/// This must not be exposed to other modules.
#[derive(Debug)]
enum IrCommand {
    // Embs
    Allocate(sdi::Allocate),
    Deallocate(sdi::Allocate),
    CopyBlock(sdi::CopyBlock),
    MaskBlock(sdi::MaskBlock),
    FillBlock(sdi::FillBlock),
    EmbedImage(sdi::EmbedImage),
    EmbedText(sdi::EmbedText),
    DecodeTokenDistribution(sdi::DecodeTokenDistribution),
    SampleTopKRequest(sdi::SampleTopKRequest),
}

#[derive(Debug)]
enum IrEvent {
    SampleTopK(sdi::SampleTopKResponse),
    GetTokenDistribution(sdi::GetTokenDistributionResponse),
}

#[derive(Debug)]
struct EventDispatcher {
    // maps correlation_id to a list of senders.
    table: HashMap<u32, Vec<EventHandle>>,
}
#[derive(Debug)]
pub enum EventHandle {
    None,
    SampleTopK(oneshot::Sender<sdi::SampleTopKResponse>),
    GetTokenDistribution(oneshot::Sender<sdi::GetTokenDistributionResponse>),
}

impl EventDispatcher {
    fn new() -> Self {
        Self {
            table: HashMap::new(),
        }
    }

    fn register(&mut self, correlation_id: u32, sender: Vec<EventHandle>) {
        self.table.insert(correlation_id, sender);
    }

    fn dispatch(&mut self, correlation_id: u32, event: Vec<IrEvent>) {
        // zip senders and evnt
        let senders = self.table.get_mut(&correlation_id).unwrap();
        assert_eq!(senders.len(), event.len());

        for (sender, evt) in senders.drain(..).zip(event.iter()) {
            match sender {
                EventHandle::None => {}
                EventHandle::SampleTopK(s) => {
                    if let IrEvent::SampleTopK(resp) = evt {
                        let _ = s.send(resp.clone());
                    } else {
                        eprintln!("Unexpected event type");
                    }
                }
                EventHandle::GetTokenDistribution(s) => {
                    if let IrEvent::GetTokenDistribution(resp) = evt {
                        let _ = s.send(resp.clone());
                    } else {
                        eprintln!("Unexpected event type");
                    }
                }
            }
        }
    }
}
// Define actual cmd, interpretable by the backend.

// Allocate(Type, entities:[id])
// Deallocate(Type, entities:[id])
// Fill (List[id,
// LongFill using bitsets.. implement this later.

#[derive(Debug, Clone)]
pub struct Backend {
    block_size: u32,
    id_pool: Arc<Mutex<ObjectIdPool>>,

    cmd_buffer: Arc<Mutex<Vec<(StreamId, IrCommand, Option<oneshot::Sender<IrEvent>>)>>>,

    pending: Arc<Mutex<Pending>>,
    staged: Arc<Mutex<Vec<sdi::Command>>>,
    submitted: Arc<Mutex<Vec<sdi::Command>>>,
    // queue, cmd_buffer, scheduled, submitted
    socket_tx: Arc<Mutex<Option<mpsc::Sender<Vec<sdi::Command>>>>>,
    handle: Arc<Mutex<Option<JoinHandle<()>>>>,
    // zmq handles
    //handle: Arc<JoinHandle<()>>,

    // event dispatcher
    event_dispatcher: Arc<Mutex<EventDispatcher>>,
}

impl Backend {
    pub fn new(block_size: u32, max_kv_blocks: u32, max_embs: u32) -> Self {
        Self {
            block_size,
            id_pool: Arc::new(Mutex::new(ObjectIdPool::new(max_kv_blocks, max_embs))),
            cmd_buffer: Arc::new(Mutex::new(Vec::new())),
            pending: Arc::new(Mutex::new(Pending::new(10.0, 1, 1))),
            staged: Arc::new(Mutex::new(Vec::new())),
            submitted: Arc::new(Mutex::new(vec![])),
            socket_tx: Arc::new(Mutex::new(None)),
            handle: Arc::new(Mutex::new(None)),
            event_dispatcher: Arc::new(Mutex::new(EventDispatcher {
                table: HashMap::new(),
            })),
        }
    }

    pub fn enqueue_cmd(&self, stream_id: StreamId, cmd: IrCommand) -> Result<(), BlockError> {
        let mut inner = self.cmd_buffer.lock().map_err(|_| BlockError::LockError)?;
        inner.push((stream_id, cmd, None));
        Ok(())
    }

    pub fn enqueue_cmd_with_resp(
        &self,
        stream_id: StreamId,
        cmd: IrCommand,
    ) -> Result<oneshot::Receiver<IrEvent>, BlockError> {
        let mut inner = self.cmd_buffer.lock().map_err(|_| BlockError::LockError)?;

        // create an oneshot channel
        let (tx, rx) = oneshot::channel();

        inner.push((stream_id, cmd, Some(tx)));
        Ok(rx)
    }

    fn acquire_id(&self, ns: ObjectNamespace) -> Result<ObjectId, BlockError> {
        let mut inner = self.id_pool.lock().map_err(|_| BlockError::LockError)?;
        inner.acquire_id(ns)
    }

    fn release_id(&self, ns: ObjectNamespace, id: ObjectId) -> Result<(), BlockError> {
        let mut inner = self.id_pool.lock().map_err(|_| BlockError::LockError)?;
        inner.release_id(ns, id)
    }

    fn remaining_ids(&self, ns: ObjectNamespace) -> usize {
        let inner = self.id_pool.lock().unwrap();
        inner.remaining_ids(ns)
    }

    pub fn schedule(&self, curr_timestamp: f64) -> Result<(), BlockError> {
        let mut pending = self.pending.lock().map_err(|_| BlockError::LockError)?;

        // first move all the commands from cmd_buffer to pending (buffer items are removed)
        let mut commands_by_stream = {
            let mut cmd_buffer = self.cmd_buffer.lock().map_err(|_| BlockError::LockError)?;

            let mut stream_commands = HashMap::new();

            for (stream_id, command, sender) in cmd_buffer.drain(..) {
                stream_commands
                    .entry(stream_id)
                    .or_insert_with(Vec::new)
                    .push((command, sender));
            }

            stream_commands
            // drop the lock on cmd_buffer
        };

        // Horizontal batching: group commands by stream and type.
        for (_stream_id, cmd_list) in commands_by_stream.iter_mut() {
            let mut prev_cmd = None;

            loop {
                if cmd_list.is_empty() {
                    break;
                }
                let (cmd, sender) = cmd_list.pop().unwrap();
                let curr_cmd = mem::discriminant(&cmd);

                // Vertical batching: Same kind of consecutive commands are batched together.
                // if the current command is different from the previous one, stop batching.
                if let Some(prev_cmd) = prev_cmd {
                    if prev_cmd != curr_cmd {
                        break;
                    }
                }

                pending.push(cmd, curr_timestamp, sender);
                prev_cmd = Some(curr_cmd);
            }
        }

        let batched_payloads = pending.batch_all(curr_timestamp);

        // add the commands to the staged queue
        let mut staged = self.staged.lock().map_err(|_| BlockError::LockError)?;

        // Add the batched commands to the staged queue.
        for (payload, senders) in batched_payloads {
            let correlation_id = self.acquire_id(ObjectNamespace::Cmd)?;

            staged.push(sdi::Command {
                correlation_id,
                payload: Some(payload),
            });

            // if at least one sender is present, add it to the event dispatcher.
            let has_senders = senders.iter().any(|s| s.is_some());
            if has_senders {
                let mut dispatcher = self
                    .event_dispatcher
                    .lock()
                    .map_err(|_| BlockError::LockError)?;
                dispatcher.table.insert(correlation_id, senders);
            }
        }

        // drop the lock on pending
        Ok(())
    }

    pub async fn commit(&self) -> Result<(), BlockError> {
        // Lock and take the staged commands.
        let mut staged = self.staged.lock().map_err(|_| BlockError::LockError)?;

        // Lock the sender.
        let tx_guard = self.socket_tx.lock().map_err(|_| BlockError::LockError)?;
        let tx = tx_guard.as_ref().ok_or(BlockError::LockError)?;

        // Send the staged commands, replacing them with an empty Vec.
        tx.send(mem::take(&mut *staged))
            .await
            .map_err(|_| BlockError::SendError)?;

        Ok(())
    }

    pub async fn bind(&mut self, endpoint: &str) -> Result<(), ZmqError> {
        // bind the zmq socket
        let mut socket = DealerSocket::new();
        socket.connect(endpoint).await?;
        println!("Connected to server at {endpoint}");

        let (tx, rx) = mpsc::channel::<Vec<sdi::Command>>(100);

        self.socket_tx = Arc::new(Mutex::new(Some(tx)));

        // 3) Spawn the single I/O driver task that handles all read/write from the socket
        let handle = tokio::spawn(Self::socket_driver(
            socket,
            rx,
            self.event_dispatcher.clone(),
        ));

        self.handle = Arc::new(Mutex::new(Some(handle)));

        Ok(())
    }

    async fn socket_driver(
        mut socket: DealerSocket,
        mut rx: mpsc::Receiver<Vec<sdi::Command>>,
        evt_dispatch: Arc<Mutex<EventDispatcher>>,
    ) {
        loop {
            tokio::select! {
                // A) Incoming requests from the channel => send to server
                maybe_req = rx.recv() => {
                    match maybe_req {
                        Some(cmds) => {
                            for cmd in cmds {
                                let bytes = cmd.encode_to_vec();
                                if let Err(e) = socket.send(ZmqMessage::from(bytes)).await {
                                    eprintln!("Socket send failed: {:?}", e);
                                    // You might choose to break or keep trying
                                }
                            }
                        },
                        None => {
                            // channel closed => no more requests
                            println!("Request channel closed, shutting down driver.");
                            break;
                        }
                    }
                },

                // B) Incoming responses from the server
                result = socket.recv() => {
                    match result {
                        Ok(msg) => {
                            // Dealer/Router typically has 2 frames: [identity, payload]
                            let payload = msg.get(0).unwrap();
                            match sdi::Event::decode(payload.as_ref()) {
                                Ok(evt) => {

                                    // send this evt somewhere elese.

                                    let correlation_id = evt.correlation_id;

                                    let ir_events = match evt.payload.unwrap() {
                                        sdi::event::Payload::SampleTopK(batch) => {
                                            batch.items.into_iter().map(|item| IrEvent::SampleTopK(item)
                                            ).collect()
                                        },
                                        sdi::event::Payload::GetTokenDistribution(batch) => {
                                             batch.items.into_iter().map(|item| IrEvent::GetTokenDistribution(item)
                                            ).collect()
                                        }
                                    };

                                    let mut dispatcher = evt_dispatch.lock().unwrap();
                                    dispatcher.dispatch(correlation_id, ir_events);

                                }
                                Err(err) => {
                                    eprintln!("Failed to parse Response from server: {:?}", err);
                                }
                            }
                        },
                        Err(e) => {
                            eprintln!("Socket receive error: {:?}", e);
                            // Possibly break or keep going...
                            break;
                        }
                    }
                }
            }
        }
    }
}

// more sophisticated forms include: MultiNodeBackend, etc.
#[derive(Debug)]
struct ObjectIdPool {
    kv_block_id_pool: IdPool<ObjectId>,
    emb_id_pool: IdPool<ObjectId>,
    dist_id_pool: IdPool<ObjectId>,
    cmd_id_pool: IdPool<ObjectId>,
}
#[derive(Debug)]
struct Pending {
    allocate: BatchQueue<sdi::Allocate>,
    deallocate: BatchQueue<sdi::Allocate>,
    copy_block: BatchQueue<sdi::CopyBlock>,
    mask_block: BatchQueue<sdi::MaskBlock>,
    embed_text: BatchQueue<sdi::EmbedText>,
    embed_image: BatchQueue<sdi::EmbedImage>,

    // these cmds are only be fired when it contains "enough" commands to be batched.
    fill_block: BatchQueue<sdi::FillBlock>,
    decode_token_distribution: BatchQueue<sdi::DecodeTokenDistribution>,
    sample_top_k: BatchQueue<sdi::SampleTopKRequest>,
}

/// "K-or-T" Strategy
// 	For instance: If queue size reaches K, launch immediately; otherwise launch after T ms if K isn’t reached.
// 	This ensures that the GPU does not stay idle for too long (bounded by T) and that short bursts of arrivals form a large enough batch to get good utilization (bounded by K).
#[derive(Debug)]
struct BatchQueue<T> {
    // cmd, timestamp, response_sender
    items: Vec<(T, f64, Option<oneshot::Sender<IrEvent>>)>,

    max_wait_time: f64,
    min_size: usize,
    max_size: usize,
}

impl<T> BatchQueue<T> {
    fn eager() -> Self {
        Self {
            items: Vec::new(),
            max_wait_time: 0.0,
            min_size: 1,
            max_size: usize::MAX,
        }
    }

    fn k_only(min_size: usize, max_size: Option<usize>) -> Self {
        Self {
            items: Vec::new(),
            max_wait_time: f64::MAX,
            min_size,
            max_size: max_size.unwrap_or(min_size),
        }
    }

    fn t_only(max_wait_time: f64) -> Self {
        Self {
            items: Vec::new(),
            max_wait_time,
            min_size: 1,
            max_size: usize::MAX,
        }
    }

    fn k_or_t(max_wait_time: f64, min_size: usize, max_size: Option<usize>) -> Self {
        Self {
            items: Vec::new(),
            max_wait_time,
            min_size,
            max_size: max_size.unwrap_or(min_size),
        }
    }

    fn take(&mut self) -> (Vec<T>, Vec<Option<oneshot::Sender<IrEvent>>>) {
        let drain_count = self.items.len().min(self.max_size);
        self.items
            .drain(..drain_count)
            .map(|(item, _, sender)| (item, sender))
            .unzip()
    }

    fn push(&mut self, item: T, curr_timestamp: f64, evt_sender: Option<oneshot::Sender<IrEvent>>) {
        self.items.push((item, curr_timestamp, evt_sender));
    }

    fn is_ready(&self, curr_timestamp: f64) -> bool {
        let num_items = self.items.len();

        if num_items > 0 {
            let longest_wait_time = curr_timestamp - self.items[0].1;
            if num_items >= self.min_size || longest_wait_time >= self.max_wait_time {
                return true;
            }
        }
        false
    }

    fn batch(
        &mut self,
        curr_timestamp: f64,
    ) -> Option<(Vec<T>, Vec<Option<oneshot::Sender<IrEvent>>>)> {
        if self.is_ready(curr_timestamp) {
            Some(self.take())
        } else {
            None
        }
    }
}

impl Pending {
    fn new(max_wait_time: f64, min_size: usize, max_size: usize) -> Self {
        Self {
            allocate: BatchQueue::eager(),
            deallocate: BatchQueue::eager(),
            copy_block: BatchQueue::k_or_t(max_wait_time, min_size, Some(max_size)),
            mask_block: BatchQueue::eager(),
            embed_text: BatchQueue::eager(),
            embed_image: BatchQueue::k_or_t(max_wait_time, min_size, Some(max_size)),
            fill_block: BatchQueue::k_or_t(max_wait_time, min_size, Some(max_size)),
            sample_top_k: BatchQueue::k_or_t(max_wait_time, min_size, Some(max_size)),
            decode_token_distribution: BatchQueue::eager(),
        }
    }

    fn push(
        &mut self,
        cmd: IrCommand,
        curr_timestamp: f64,
        evt_sender: Option<oneshot::Sender<IrEvent>>,
    ) {
        match cmd {
            IrCommand::Allocate(item) => {
                self.allocate.push(item, curr_timestamp, evt_sender);
            }
            IrCommand::Deallocate(item) => {
                self.deallocate.push(item, curr_timestamp, evt_sender);
            }
            IrCommand::CopyBlock(item) => {
                self.copy_block.push(item, curr_timestamp, evt_sender);
            }
            IrCommand::MaskBlock(item) => {
                self.mask_block.push(item, curr_timestamp, evt_sender);
            }
            IrCommand::FillBlock(item) => {
                self.fill_block.push(item, curr_timestamp, evt_sender);
            }
            IrCommand::EmbedImage(item) => {
                self.embed_image.push(item, curr_timestamp, evt_sender);
            }
            IrCommand::EmbedText(item) => {
                self.embed_text.push(item, curr_timestamp, evt_sender);
            }
            IrCommand::SampleTopKRequest(item) => {
                self.sample_top_k.push(item, curr_timestamp, evt_sender);
            }
            IrCommand::DecodeTokenDistribution(item) => {
                self.decode_token_distribution
                    .push(item, curr_timestamp, evt_sender);
            }
        }
    }

    fn batch_all(
        &mut self,
        curr_timestamp: f64,
    ) -> Vec<(sdi::command::Payload, Vec<Option<oneshot::Sender<IrEvent>>>)> {
        let mut cmds = Vec::new();

        if let Some((items, senders)) = self.allocate.batch(curr_timestamp) {
            cmds.push((
                sdi::command::Payload::Allocate(sdi::BatchAllocate { items }),
                senders,
            ));
        }

        if let Some((items, senders)) = self.deallocate.batch(curr_timestamp) {
            cmds.push((
                sdi::command::Payload::Deallocate(sdi::BatchDeallocate { items }),
                senders,
            ));
        }

        if let Some((items, senders)) = self.copy_block.batch(curr_timestamp) {
            cmds.push((
                sdi::command::Payload::CopyBlock(sdi::BatchCopyBlock { items }),
                senders,
            ));
        }

        if let Some((items, senders)) = self.mask_block.batch(curr_timestamp) {
            cmds.push((
                sdi::command::Payload::MaskBlock(sdi::BatchMaskBlock { items }),
                senders,
            ));
        }

        if let Some((items, senders)) = self.embed_text.batch(curr_timestamp) {
            cmds.push((
                sdi::command::Payload::EmbedText(sdi::BatchEmbedText { items }),
                senders,
            ));
        }

        if let Some((items, senders)) = self.embed_image.batch(curr_timestamp) {
            cmds.push((
                sdi::command::Payload::EmbedImage(sdi::BatchEmbedImage { items }),
                senders,
            ));
        }

        if let Some((items, senders)) = self.fill_block.batch(curr_timestamp) {
            cmds.push((
                sdi::command::Payload::FillBlock(sdi::BatchFillBlock { items }),
                senders,
            ));
        }

        if let Some((items, senders)) = self.sample_top_k.batch(curr_timestamp) {
            cmds.push((
                sdi::command::Payload::SampleTopKRequest(sdi::BatchSampleTopKRequest { items }),
                senders,
            ));
        }

        cmds
    }
}

enum ObjectNamespace {
    KvBlock = 0,
    Emb = 1,
    Dist = 2,
    Cmd = 3,
}

impl ObjectIdPool {
    fn new(max_kv_blocks: u32, max_embs: u32) -> Self {
        Self {
            kv_block_id_pool: IdPool::new(max_kv_blocks),
            emb_id_pool: IdPool::new(max_embs),
            dist_id_pool: IdPool::new(max_embs),
            cmd_id_pool: IdPool::new(max_embs),
        }
    }

    fn acquire_id(&mut self, ns: ObjectNamespace) -> Result<ObjectId, BlockError> {
        match ns {
            ObjectNamespace::KvBlock => self.kv_block_id_pool.acquire(),
            ObjectNamespace::Emb => self.emb_id_pool.acquire(),
            ObjectNamespace::Dist => self.dist_id_pool.acquire(),
            ObjectNamespace::Cmd => self.cmd_id_pool.acquire(),
        }
        .ok_or(BlockError::NoFreeBlocks)
    }

    fn release_id(&mut self, ns: ObjectNamespace, id: ObjectId) -> Result<(), BlockError> {
        match ns {
            ObjectNamespace::KvBlock => self.kv_block_id_pool.release(id),
            ObjectNamespace::Emb => self.emb_id_pool.release(id),
            ObjectNamespace::Dist => self.dist_id_pool.release(id),
            ObjectNamespace::Cmd => self.cmd_id_pool.release(id),
        }
    }

    fn remaining_ids(&self, ns: ObjectNamespace) -> usize {
        match ns {
            ObjectNamespace::KvBlock => self.kv_block_id_pool.available(),
            ObjectNamespace::Emb => self.emb_id_pool.available(),
            ObjectNamespace::Dist => self.dist_id_pool.available(),
            ObjectNamespace::Cmd => self.cmd_id_pool.available(),
        }
    }
}

impl ObjectAllocator<TokenEmb> for Backend {
    type RawRepr = Vec<f32>;

    fn alloc(&self, stream_id: StreamId) -> Result<ObjectId, BlockError> {
        let new_obj_id = self.acquire_id(ObjectNamespace::Emb)?;

        let cmd = IrCommand::Allocate(sdi::Allocate {
            kind: sdi::ObjectKind::Emb.into(),
            object_id_offset: new_obj_id,
            count: 1,
        });
        self.enqueue_cmd(stream_id, cmd)?;

        Ok(new_obj_id)
    }

    fn dealloc(&self, stream_id: StreamId, obj_id: ObjectId) -> Result<(), BlockError> {
        // Release the object id back to the pool.
        self.release_id(ObjectNamespace::Emb, obj_id)?;

        let cmd = IrCommand::Deallocate(sdi::Allocate {
            kind: sdi::ObjectKind::Emb.into(),
            object_id_offset: obj_id,
            count: 1,
        });
        self.enqueue_cmd(stream_id, cmd)?;

        Ok(())
    }

    fn raw_repr(
        &self,
        stream_id: StreamId,
        obj_id: ObjectId,
    ) -> Result<oneshot::Receiver<Self::RawRepr>, BlockError> {
        todo!()
    }

    fn available(&self) -> usize {
        self.remaining_ids(ObjectNamespace::Emb)
    }
}

impl ObjectAllocator<KvBlock> for Backend {
    // The raw representation of a kv block is not really useful in any way. So we just use usize.
    type RawRepr = ObjectId;

    fn alloc(&self, stream_id: StreamId) -> Result<ObjectId, BlockError> {
        let new_obj_id = self.acquire_id(ObjectNamespace::KvBlock)?;

        let cmd = IrCommand::Allocate(sdi::Allocate {
            kind: sdi::ObjectKind::KvBlock.into(),
            object_id_offset: new_obj_id,
            count: 1,
        });
        self.enqueue_cmd(stream_id, cmd)?;

        Ok(new_obj_id)
    }

    fn dealloc(&self, stream_id: StreamId, obj_id: ObjectId) -> Result<(), BlockError> {
        // Release the object id back to the pool.
        self.release_id(ObjectNamespace::KvBlock, obj_id)?;

        let cmd = IrCommand::Deallocate(sdi::Allocate {
            kind: sdi::ObjectKind::KvBlock.into(),
            object_id_offset: obj_id,
            count: 1,
        });
        self.enqueue_cmd(stream_id, cmd)?;

        Ok(())
    }

    fn raw_repr(
        &self,
        stream_id: StreamId,
        obj_id: ObjectId,
    ) -> Result<oneshot::Receiver<Self::RawRepr>, BlockError> {
        //sender.send(obj_id).unwrap();
        Ok(())
    }

    fn available(&self) -> usize {
        self.remaining_ids(ObjectNamespace::KvBlock)
    }
}

impl ObjectAllocator<TokenDist> for Backend {
    type RawRepr = Vec<f32>;

    fn alloc(&self, stream_id: StreamId) -> Result<ObjectId, BlockError> {
        todo!()
    }

    fn dealloc(&self, stream_id: StreamId, obj_id: ObjectId) -> Result<(), BlockError> {
        todo!()
    }

    fn raw_repr(
        &self,
        stream_id: StreamId,
        obj_id: ObjectId,
    ) -> Result<oneshot::Receiver<Self::RawRepr>, BlockError> {
        todo!()
    }

    fn available(&self) -> usize {
        todo!()
    }
}

impl CausalTransformer for Backend {
    fn fill(
        &self,
        stream_id: StreamId,
        ptr: ObjectId,
        ctx_ptrs: Vec<ObjectId>,
        input_embs: Vec<ObjectId>,
        output_embs: Option<Vec<ObjectId>>,
    ) -> Result<(), BlockError> {
        // create "resolved" cmd.

        let cmd = IrCommand::FillBlock(sdi::FillBlock {
            block_id: ptr,
            context_block_ids: ctx_ptrs,
            input_embedding_ids: input_embs,
            output_embedding_ids: output_embs.unwrap_or_default(),
        });

        self.enqueue_cmd(stream_id, cmd)?;
        Ok(())
    }

    fn copy_tokens(
        &self,
        stream_id: StreamId,
        src_ptr: ObjectId,
        dst_ptr: ObjectId,
        src_offset: u32,
        dst_offset: u32,
        size: u32,
    ) -> Result<(), BlockError> {
        let cmd = IrCommand::CopyBlock(sdi::CopyBlock {
            source_block_id: src_ptr,
            destination_block_id: dst_ptr,
            source_start: src_offset,
            destination_start: dst_offset,
            length: size,
        });

        self.enqueue_cmd(stream_id, cmd)?;
        Ok(())
    }

    fn mask_tokens(
        &self,
        stream_id: StreamId,
        ptr: ObjectId,
        mask: &[bool],
    ) -> Result<(), BlockError> {
        let cmd = IrCommand::MaskBlock(sdi::MaskBlock {
            block_id: ptr,
            mask: mask.to_vec(),
        });
        self.enqueue_cmd(stream_id, cmd)?;
        Ok(())
    }
}

impl CausalLanguageModel for Backend {
    fn next_token_dist(
        &self,
        stream_id: StreamId,
        emb_ptr: ObjectId,
        dist_ptr: ObjectId,
    ) -> Result<(), BlockError> {
        todo!()
    }

    fn sample_top_k(
        &self,
        stream_id: StreamId,
        dist_ptr: ObjectId,
        k: usize,
    ) -> Result<oneshot::Receiver<Vec<u32>>, BlockError> {
        let (resp_tx, resp_rx) = oneshot::channel::<sdi::DecodeResponse>();

        let rx_chain = resp_rx.and_then(|resp| async move {
            // Process the number (e.g., add a prefix)
            // Send it using tx2; since tx2.send(...) returns a Result,
            // we wrap it with future::result to lift it into a future.
            Ok(resp.token_ids[0])
        });

        todo!()
    }

    fn get_raw_dist(
        &self,
        stream_id: StreamId,
        dist_ptr: ObjectId,
    ) -> Result<oneshot::Receiver<Vec<f32>>, BlockError> {
        todo!()
    }
}

/// for multimodal LLMs

impl ImageEmbedder for Backend {
    fn embed_img(
        &self,
        stream_id: StreamId,
        addrs: Vec<ObjectId>,
        url: String,
    ) -> Result<(), BlockError> {
        todo!()
    }
}
