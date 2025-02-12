use crate::state::{
    BlockError, CausalLanguageModel, CausalTransformer, ImageEmbedder, KvBlock, ObjectAllocator,
    RemoteObjId, StreamId, TokenDist, TokenEmb,
};
use crate::utils::IdPool;
use prost::Message;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::oneshot::Sender;

pub mod sdi {
    include!(concat!(env!("OUT_DIR"), "/sdi.rs"));
}

/// Intermediate representation of a command to be executed by the backend.
/// This must not be exposed to other modules.
#[derive(Debug)]
enum IrCommand {
    // Embs
    AllocateEmb(usize),
    DeallocateEmb(usize),

    // KvBlocks
    AllocateKvBlock(usize),
    DeallocateKvBlock(usize),
    CopyKvBlock(usize, usize, usize, usize, usize),
    MaskKvBlock(usize, Vec<bool>),
    FillKvBlock(usize, Vec<usize>, Vec<usize>, Option<Vec<usize>>),
}

// Define actual cmd, interpretable by the backend.

// Allocate(Type, entities:[id])
// Deallocate(Type, entities:[id])
// Fill (List[id,
// LongFill using bitsets.. implement this later.

#[derive(Debug, Clone)]
pub struct Backend {
    block_size: usize,
    namespace: Arc<Mutex<ObjNamespace>>,

    cmd_buffer: Arc<Mutex<Vec<(StreamId, IrCommand)>>>,

    pending: Arc<Mutex<Pending>>,
    staged: Arc<Mutex<HashMap<StreamId, IrCommand>>>,
    submitted: Arc<Mutex<Vec<(StreamId, IrCommand)>>>,
    // queue, cmd_buffer, scheduled, submitted
}

// more sophisticated forms include: MultiNodeBackend, etc.
#[derive(Debug)]
struct ObjNamespace {
    kv_block_id_pool: IdPool<RemoteObjId>,
    emb_id_pool: IdPool<RemoteObjId>,
}
#[derive(Debug)]
struct Pending {
    fill_cmds: Vec<IrCommand>,
}
impl Pending {
    fn new() -> Self {
        Self {
            fill_cmds: Vec::new(),
        }
    }
}

impl ObjNamespace {
    fn new(max_kv_blocks: usize, max_embs: usize) -> Self {
        Self {
            kv_block_id_pool: IdPool::new(max_kv_blocks),
            emb_id_pool: IdPool::new(max_embs),
        }
    }

    fn acquire_id(&mut self, namespace: usize) -> Result<RemoteObjId, BlockError> {
        match namespace {
            0 => self.kv_block_id_pool.acquire(),
            1 => self.emb_id_pool.acquire(),
            _ => return Err(BlockError::VirtualAddressTranslationFailed),
        }
        .ok_or(BlockError::NoFreeBlocks)
    }

    fn release_id(&mut self, namespace: usize, id: RemoteObjId) -> Result<(), BlockError> {
        match namespace {
            0 => self.kv_block_id_pool.release(id),
            1 => self.emb_id_pool.release(id),
            _ => Err(BlockError::VirtualAddressTranslationFailed),
        }
    }

    fn remaining_ids(&self, namespace: usize) -> usize {
        match namespace {
            0 => self.kv_block_id_pool.available(),
            1 => self.emb_id_pool.available(),
            _ => 0,
        }
    }
}

impl Backend {
    pub fn new(block_size: usize, max_kv_blocks: usize, max_embs: usize) -> Self {
        Self {
            block_size,
            namespace: Arc::new(Mutex::new(ObjNamespace::new(max_kv_blocks, max_embs))),
            cmd_buffer: Arc::new(Mutex::new(Vec::new())),
            pending: Arc::new(Mutex::new(Pending::new())),
            staged: Arc::new(Mutex::new(HashMap::new())),
            submitted: Arc::new(Mutex::new(vec![])),
        }
    }

    pub fn enqueue_cmd(&self, stream_id: StreamId, cmd: IrCommand) -> Result<(), BlockError> {
        let mut inner = self.cmd_buffer.lock().map_err(|_| BlockError::LockError)?;
        inner.push((stream_id, cmd));
        Ok(())
    }

    fn acquire_id(&self, namespace: usize) -> Result<RemoteObjId, BlockError> {
        let mut inner = self.namespace.lock().map_err(|_| BlockError::LockError)?;
        inner.acquire_id(namespace)
    }

    fn release_id(&self, namespace: usize, id: RemoteObjId) -> Result<(), BlockError> {
        let mut inner = self.namespace.lock().map_err(|_| BlockError::LockError)?;
        inner.release_id(namespace, id)
    }

    fn remaining_ids(&self, namespace: usize) -> usize {
        let inner = self.namespace.lock().unwrap();
        inner.remaining_ids(namespace)
    }

    pub fn schedule(&self, time_elapsed: f64) -> Result<(), BlockError> {
        let mut pending = self.pending.lock().map_err(|_| BlockError::LockError)?;

        // first move all the commands from cmd_buffer to pending (buffer items are removed)
        let mut commands_by_stream = {
            let mut cmd_buffer = self.cmd_buffer.lock().map_err(|_| BlockError::LockError)?;

            let mut stream_commands = HashMap::new();

            for (stream_id, command) in cmd_buffer.drain(..) {
                stream_commands
                    .entry(stream_id)
                    .or_insert_with(Vec::new)
                    .push(command);
            }

            stream_commands
            // drop the lock on cmd_buffer
        };

        // do batching!!
        //
        // cmd_set = HashMap<CmdType, Vec<Cmd>>
        //
        let mut cmd_alloc_emb = Vec::new();
        let mut cmd_alloc_kv_block = Vec::new();
        let mut cmd_dealloc_emb = Vec::new();
        let mut cmd_dealloc_kv_block = Vec::new();

        for (_stream_id, cmd_list) in commands_by_stream.iter_mut() {
            let mut processed_count = 0;

            for cmd in cmd_list.iter() {
                let mut should_stop = true;
                match cmd {
                    IrCommand::AllocateEmb(id) => {
                        cmd_alloc_emb.push(*id);
                        // Processed, so keep_on remains false (we can drop this command)
                    }
                    IrCommand::DeallocateEmb(id) => {
                        cmd_dealloc_emb.push(*id);
                    }
                    IrCommand::AllocateKvBlock(id) => {
                        cmd_alloc_kv_block.push(*id);
                    }
                    IrCommand::DeallocateKvBlock(id) => {
                        cmd_dealloc_kv_block.push(*id);
                    }
                    IrCommand::CopyKvBlock(src, dst, src_offset, dst_offset, size) => {

                        // println!(
                        //     "Copying kv block from {} to {} with offsets {} and {} and size {}",
                        //     src, dst, src_offset, dst_offset, size
                        // );
                    }
                    IrCommand::MaskKvBlock(id, mask) => {
                        println!("Masking kv block with id: {} and mask: {:?}", id, mask);
                    }
                    IrCommand::FillKvBlock(ptr, ctx_ptrs, input_embs, output_embs) => {
                        println!(
                            "Filling kv block with ptr: {}, ctx_ptrs: {:?}, input_embs: {:?}, output_embs: {:?}",
                            ptr, ctx_ptrs,input_embs, output_embs
                        );
                        should_stop = false;
                    }
                }
                processed_count += 1;
                if should_stop {
                    break;
                }
            }

            // Only remove the commands that were processed.
            cmd_list.drain(0..processed_count);
        }

        if cmd_alloc_emb.len() > 0 {

            sdi::Allocate {
                kind: sdi::ObjectKind::Emb.into(),
                object_ids: Vec::new(),
            };
            // sdi::Allocate {
            //
            // };

            println!("Allocating embeddings: {:?}", cmd_alloc_emb);
        }

        // drop the lock on pending
        Ok(())
    }
}

impl ObjectAllocator<TokenEmb> for Backend {
    type RawRepr = Vec<f32>;

    fn alloc(&self, stream_id: StreamId) -> Result<RemoteObjId, BlockError> {
        let new_obj_id = self.acquire_id(1)?;

        let cmd = IrCommand::AllocateEmb(new_obj_id);
        self.enqueue_cmd(stream_id, cmd)?;

        Ok(new_obj_id)
    }

    fn dealloc(&self, stream_id: StreamId, obj_id: RemoteObjId) -> Result<(), BlockError> {
        // Release the object id back to the pool.
        self.release_id(1, obj_id)?;

        let cmd = IrCommand::DeallocateEmb(obj_id);
        self.enqueue_cmd(stream_id, cmd)?;

        Ok(())
    }

    fn raw_repr(
        &self,
        stream_id: StreamId,
        obj_id: RemoteObjId,
        sender: Sender<Self::RawRepr>,
    ) -> Result<(), BlockError> {
        todo!()
    }

    fn available(&self) -> usize {
        self.remaining_ids(1)
    }
}

impl ObjectAllocator<KvBlock> for Backend {
    // The raw representation of a kv block is not really useful in any way. So we just use usize.
    type RawRepr = usize;

    fn alloc(&self, stream_id: StreamId) -> Result<RemoteObjId, BlockError> {
        let new_obj_id = self.acquire_id(0)?;

        let cmd = IrCommand::AllocateKvBlock(new_obj_id);
        self.enqueue_cmd(stream_id, cmd)?;

        Ok(new_obj_id)
    }

    fn dealloc(&self, stream_id: StreamId, obj_id: RemoteObjId) -> Result<(), BlockError> {
        // Release the object id back to the pool.
        self.release_id(0, obj_id)?;

        let cmd = IrCommand::DeallocateKvBlock(obj_id);
        self.enqueue_cmd(stream_id, cmd)?;

        Ok(())
    }

    fn raw_repr(
        &self,
        stream_id: StreamId,
        obj_id: RemoteObjId,
        sender: Sender<Self::RawRepr>,
    ) -> Result<(), BlockError> {
        sender.send(obj_id).unwrap();
        Ok(())
    }

    fn available(&self) -> usize {
        self.remaining_ids(0)
    }
}

impl ObjectAllocator<TokenDist> for Backend {
    type RawRepr = Vec<f32>;

    fn alloc(&self, stream_id: StreamId) -> Result<RemoteObjId, BlockError> {
        todo!()
    }

    fn dealloc(&self, stream_id: StreamId, obj_id: RemoteObjId) -> Result<(), BlockError> {
        todo!()
    }

    fn raw_repr(
        &self,
        stream_id: StreamId,
        obj_id: RemoteObjId,
        sender: Sender<Self::RawRepr>,
    ) -> Result<(), BlockError> {
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
        ptr: RemoteObjId,
        ctx_ptrs: Vec<RemoteObjId>,
        input_embs: Vec<RemoteObjId>,
        output_embs: Option<Vec<RemoteObjId>>,
    ) -> Result<(), BlockError> {
        // create "resolved" cmd.

        let cmd = IrCommand::FillKvBlock(ptr, ctx_ptrs, input_embs, output_embs);

        self.enqueue_cmd(stream_id, cmd)?;
        Ok(())
    }

    fn copy_tokens(
        &self,
        stream_id: StreamId,
        src_ptr: RemoteObjId,
        dst_ptr: RemoteObjId,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> Result<(), BlockError> {
        let cmd = IrCommand::CopyKvBlock(src_ptr, dst_ptr, src_offset, dst_offset, size);

        self.enqueue_cmd(stream_id, cmd)?;
        Ok(())
    }

    fn mask_tokens(
        &self,
        stream_id: StreamId,
        ptr: RemoteObjId,
        mask: &[bool],
    ) -> Result<(), BlockError> {
        let cmd = IrCommand::MaskKvBlock(ptr, mask.to_vec());
        self.enqueue_cmd(stream_id, cmd)?;
        Ok(())
    }
}

impl CausalLanguageModel for Backend {
    fn next_token_dist(
        &self,
        stream_id: StreamId,
        emb_ptr: RemoteObjId,
        dist_ptr: RemoteObjId,
    ) -> Result<(), BlockError> {
        todo!()
    }

    fn sample_top_k(
        &self,
        stream_id: StreamId,
        dist_ptr: RemoteObjId,
        k: usize,
        sender: Sender<Vec<usize>>,
    ) -> Result<(), BlockError> {
        todo!()
    }

    fn get_raw_dist(
        &self,
        stream_id: StreamId,
        dist_ptr: RemoteObjId,
        sender: Sender<Vec<f32>>,
    ) -> Result<(), BlockError> {
        todo!()
    }
}

/// for multimodal LLMs

impl ImageEmbedder for Backend {
    fn embed_img(
        &self,
        stream_id: StreamId,
        addrs: Vec<RemoteObjId>,
        url: String,
    ) -> Result<(), BlockError> {
        todo!()
    }
}
