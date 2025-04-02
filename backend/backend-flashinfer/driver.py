import time
from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
from transformers import AutoTokenizer

import config
from common import ceil_div
from l4ma import AttentionStorage, VectorStorage
from llama import LlamaForCausalLM
from l4m_pb2 import BatchAllocate, BatchDeallocate, BatchEmbedText, BatchMaskBlock, BatchCopyBlock, BatchDecodeTokenDistribution, BatchSampleTopKRequest, BatchSampleTopKResponse, \
    ObjectKind, SampleTopKResponse, BatchFillBlock

from l4m_vision_pb2 import BatchEmbedImage
from config import FULL_MODEL_NAME, NUM_TOKENS_IN_BLOCK

tokenizer = AutoTokenizer.from_pretrained(FULL_MODEL_NAME)


@dataclass
class TextEmbed:
    token_id: int
    position_id: int


@dataclass
class ImageEmbed:
    vec_id: int
    position_id: tuple[int, int]


@dataclass
class Block:
    position_ids: np.ndarray
    occupancy: np.ndarray


EMPTY_BLOCK = Block(
    position_ids=np.array([0] * NUM_TOKENS_IN_BLOCK),
    occupancy=np.array([False] * NUM_TOKENS_IN_BLOCK)
)

# union type
Embed = Union[TextEmbed, ImageEmbed]


class Driver:
    embeds: dict[int, Embed]
    blocks: dict[int, Block]
    lm: LlamaForCausalLM


    # dist_storage: VectorStorage

    def __init__(self, model, max_num_pages: int, dtype: torch.dtype, device: str):
        self.embeds = {}
        self.blocks = {}

        self.lm = model
        self.max_num_pages = max_num_pages
        self.dtype = dtype
        self.device = device

        self.kv_cache_at_layer = [
            torch.randn(
                (max_num_pages, 2, NUM_TOKENS_IN_BLOCK,
                self.lm.config.num_key_value_heads,
                self.lm.config.hidden_size // self.lm.config.num_attention_heads),
                dtype=dtype, device=device
            ) for _ in range(self.lm.config.num_hidden_layers)
        ]

        self.embed_storage_p1 = VectorStorage(
            num_vectors=50000,
            embed_dim=config.DIST_RESOLUTION,
            dtype=dtype,
            device=device
        )

        self.embed_storage_p2 = VectorStorage(
            num_vectors=50000,
            embed_dim=config.DIST_RESOLUTION,
            dtype=torch.long,
            device=device
        )

        # self.dist_storage = dist_storage

        self.inter_fill_time = time.time()


    def allocate(self, cmds: BatchAllocate):
        # in current implementation, all allocations are already done in the constructor.
        # but in the future, we may want to allocate more blocks than the GPU capacity, by offloading some of the blocks to the CPU memory.
        # This logic should handle that case.

        for cmd in cmds.items:
            if cmd.kind == ObjectKind.OBJECT_KIND_KV_BLOCK:
                for i in range(cmd.count):
                    self.blocks[cmd.object_id_offset + i] = Block(
                        position_ids=np.array([0] * NUM_TOKENS_IN_BLOCK),
                        occupancy=np.array([False] * NUM_TOKENS_IN_BLOCK)
                    )

            elif cmd.kind == ObjectKind.OBJECT_KIND_EMB:
                # do nothing. Embeds are allocated on the fly.
                ...
            elif cmd.kind == ObjectKind.OBJECT_KIND_DIST:
                # do nothing. Dists are allocated on the fly.
                ...

    def deallocate(self, cmds: BatchDeallocate):
        # in current implementation, all allocations are already done in the constructor.
        # so we don't need to deallocate anything.
        ...

    def embed_text(self, cmds: BatchEmbedText):
        for cmd in cmds.items:
            self.embeds[cmd.embedding_id] = TextEmbed(token_id=cmd.token_id, position_id=cmd.position_id)

    def embed_image(self, cmds: BatchEmbedImage):
        # unimplemented
        ...

    def mask_block(self, cmds: BatchMaskBlock):
        for cmd in cmds.items:
            block = self.blocks[cmd.block_id]
            for i, m in enumerate(cmd.mask):
                block.occupancy[i] = m

    def copy_block(self, cmds: BatchCopyBlock):

        # TODO.
        ...

    @torch.inference_mode()
    def decode_token_distribution(self, cmds: BatchDecodeTokenDistribution):
        # start_time = time.time()

        # TODO -> make this more efficient by batching.
        # for i, cmd in enumerate(cmds.items):
        #     dist = torch.softmax(self.lm.lm_head(self.embed_storage.ptr[cmd.embedding_id]), dim=-1)
        #     self.dist_storage.ptr[cmd.distribution_id] = dist

        # torch.cuda.synchronize()
        # print(f"decode_token_distribution elapsed time {(time.time() - start_time) * 1000}ms")
        ...

    @torch.inference_mode()
    def sample_top_k_request(self, cmds: BatchSampleTopKRequest) -> BatchSampleTopKResponse:

        start_time = time.time()

        res = []
                
        for i, cmd in enumerate(cmds.items):
            topk_probs = self.embed_storage_p1.ptr[cmd.distribution_id].tolist()
            topk_tokens = self.embed_storage_p2.ptr[cmd.distribution_id].tolist()

            # topk_res = torch.topk(dist, k=cmd.k)
            # topk_tokens = topk_res.indices.tolist()
            # topk_probs = topk_res.values.tolist()

            res.append(SampleTopKResponse(token_ids=topk_tokens, probabilities=topk_probs))

        #torch.cuda.synchronize()
        #print(f"sample_top_k_request elapsed time {(time.time() - start_time) * 1000}ms")

        return BatchSampleTopKResponse(items=res)

    @torch.inference_mode()
    def fill_block(self, cmds: BatchFillBlock):


        start_time = time.time()
        
        kv_page_indices = []
        kv_page_indptr = [0]
        kv_last_page_lens = []
        qo_indices = []
        qo_indptr = [0]
        custom_masks = []
        
        new_token_ids = []
        new_position_ids = []
        output_embed_postproc = []
        single_token_inference_mode = True
        
        for i, cmd in enumerate(cmds.items):
            offset = cmd.block_id # change this name to "offset" later.
            
            ctx_block_ids = cmd.context_block_ids # block == page. make names consistent later.
            input_embeds = cmd.input_embedding_ids
            output_embeds = cmd.output_embedding_ids
            
            kv_page_indices.extend(ctx_block_ids)
            kv_page_indptr.append(len(kv_page_indices))
            kv_last_page_lens.append(offset + len(input_embeds))
            
            if offset + len(input_embeds) > NUM_TOKENS_IN_BLOCK:
                # should never happen.
                raise ValueError("Page size exceeded")
            
            qo_indices.extend(input_embeds)
            qo_indptr.append(len(qo_indices))
        
            # let's compute the mask.
            
            ctx_pos_ids = np.hstack([self.blocks[ctx_id].position_ids for ctx_id in ctx_block_ids])  # int
            ctx_occupancy = np.hstack([self.blocks[ctx_id].occupancy for ctx_id in ctx_block_ids])  # bool
            inp_pos_ids = np.empty((len(input_embeds), ), dtype=np.int32)
            inp_occupancy = np.zeros((len(input_embeds), ), dtype=np.bool_)

            if len(input_embeds) > 1:
                single_token_inference_mode = False
            
            for i in range(len(input_embeds)):
                if input_embeds[i] in self.embeds:
                    embed = self.embeds[input_embeds[i]]
                    if isinstance(embed, TextEmbed):
                        new_token_ids.append(embed.token_id)
                        new_position_ids.append(embed.position_id)
                        inp_occupancy[i] = True
                        inp_pos_ids[i] = embed.position_id
                else:
                    # should never happen, since the controller should have already checked that the input embeds are valid.
                    raise ValueError("Input embedding not found")
            
            for i in range(len(output_embeds)):
                output_embed_postproc.append({
                    "idx":  len(new_token_ids) -  len(output_embeds) + i,
                    "vec_id": output_embeds[i]
                })
            
            casual_mask = ctx_pos_ids[None, :] <= inp_pos_ids[:, None]
            valid_mask = np.logical_and(ctx_occupancy[None, :], inp_occupancy[:, None])
            mask = np.logical_and(casual_mask, valid_mask)
            
            mask_flat = mask.flatten()
            custom_masks.append(mask_flat)

        # concat all masks
        custom_mask = np.concatenate(custom_masks)

        pt_new_token_ids = torch.as_tensor(new_token_ids, device=self.device(), dtype=torch.int32)
        pt_new_position_ids = torch.as_tensor(new_position_ids, device=self.device(), dtype=torch.int32)

        pt_kv_page_indices = torch.as_tensor(kv_page_indices, device=self.device(), dtype=torch.int32)
        pt_kv_page_indptr = torch.as_tensor(kv_page_indptr, device=self.device(), dtype=torch.int32)
        pt_kv_last_page_lens = torch.as_tensor(kv_last_page_lens, device=self.device(), dtype=torch.int32)
        pt_qo_indptr = torch.as_tensor(qo_indptr, device=self.device(), dtype=torch.int32)
        pt_custom_mask = torch.as_tensor(custom_mask, device=self.device(), dtype=torch.bool_)

        input_embeds = self.lm.model.embed_tokens(pt_new_token_ids)

        torch.cuda.synchronize()
        print(f"prepare time {(time.time() - start_time) * 1000}ms  ")
        
        with torch.cuda.device(self.device()):
            output_embeds = self.lm.model.forward(
                input_embeds=input_embeds,
                position_ids=pt_new_position_ids,
                kv_cache_at_layer=self.kv_cache_at_layer,
                kv_page_indices=pt_kv_page_indices,
                kv_page_indptr=pt_kv_page_indptr,
                kv_last_page_lens=pt_kv_last_page_lens,
                qo_indptr=pt_qo_indptr,
                custom_mask=pt_custom_mask,
                single_token_inference_mode=single_token_inference_mode,
            )

            # precompute the dists
            logits = self.lm.lm_head(output_embeds)

            # topk
            condensed = torch.topk(logits, k=config.DIST_RESOLUTION, sorted=True)

        # print(logits.shape)
        # store the logits in the output embeds  -> replace with torch.scatter later
        for token_map in output_embed_postproc:
            vec_id = token_map["vec_id"]
            idx = token_map["idx"]

            self.embed_storage_p1.ptr[vec_id].copy_(condensed.values[idx], non_blocking=True)
            self.embed_storage_p2.ptr[vec_id].copy_(condensed.indices[idx], non_blocking=True)


        torch.cuda.synchronize()
        elapsed_time = (time.time() - start_time) * 1000
        print(f"fill_block elapsed time {elapsed_time}ms size {output_embeds.shape}")

        self.inter_fill_time = time.time()
