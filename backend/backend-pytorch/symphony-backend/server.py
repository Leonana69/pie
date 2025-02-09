from __future__ import annotations

import threading
import typing
from dataclasses import dataclass
from enum import StrEnum
from typing import Union
import numpy as np
import torch

from blocks import KvBlockManager, BlockStorage, KvBlock, Address, Address, EmbeddingManager, EmbeddingStorage, KvBlockStorage

type InstanceId = bytes


@dataclass
class TextToken:
    token_id: int
    position_id: int


@dataclass
class ImageToken:
    embed_addr: Address
    position_id: tuple[int, int, int]


@dataclass
class VideoToken:
    embed_addr: Address
    position_id: tuple[int, int, int]


Token = Union[TextToken, ImageToken, VideoToken]


# =============================================================================
# 1) Enums & Dataclasses for Commands/Responses
# =============================================================================

class CommandKind(StrEnum):
    # Block-level operations
    ALLOCATE_KV_BLOCKS = "AllocateKvBlocks"
    DEALLOCATE_KV_BLOCKS = "FreeKvBlocks"
    AVAILABLE_KV_BLOCKS = "AvailableKvBlocks"

    # Token-level operations
    COPY_KV_BLOCK = "CopyTokens"
    MASK_KV_BLOCK = "DropTokens"
    FILL_KV_BLOCK = "FillKvBlock"

    # Embedding vector operations (input/output)
    ALLOCATE_TOKEN_EMBEDS = "AllocateTokenEmbeds"
    DEALLOCATE_TOKEN_EMBEDS = "DeallocateTokenEmbeds"
    AVAILABLE_TOKEN_EMBEDS = "AvailableTokenEmbeds"

    #  Input embedding operations
    EMBED_IMAGE = "EmbedImage"
    EMBED_VIDEO = "EmbedVideo"

    # Output embedding operations
    GET_NEXT_TOKEN_DIST = "GetNextTokenDist"
    GET_FEATURE_VECTOR = "GetFeatureVector"
    DECODE = "Decode"

    # Add more if needed


######## BLOCK ALLOC/DEALLOC ########

@dataclass
class AllocateKvBlocksCmd:
    num_blocks: int


@dataclass
class FreeKvBlocksCmd:
    addr_offset: int
    count: int


@dataclass
class AvailableKvBlocksCmd:
    pass


######## BLOCK CTRL ########

@dataclass
class FillKvBlockCmd:
    addr: Address
    ctx_addrs: list[Address]
    mask: list[bool]

    input_embeds: list[Token]
    output_embeds: list[Address]
    output_embed_offset: int


@dataclass
class CopyKvBlockCmd:
    src_addr: int
    dst_addr: int
    src_offset: int
    dst_offset: int
    size: int


@dataclass
class MaskKvBlockCmd:
    addr: int
    offset: int
    size: int


######## BLOCK ALLOC/DEALLOC ########

@dataclass
class AllocateTokenEmbedsCmd:
    num_embeds: int


@dataclass
class DeallocateTokenEmbedsCmd:
    addr_offset: int
    count: int


@dataclass
class AvailableTokenEmbedsCmd:
    ...


#### INPUT EMBEDDING ####

@dataclass
class EmbedImageCmd:
    image_url: str


@dataclass
class EmbedVideoCmd:
    video_url: str


#### OUTPUT EMBEDDING ####


@dataclass
class DecodeCmd:
    ...


@dataclass
class GetNextTokenDistCmd:
    addr: int
    offset: int
    size: int
    drop_output_embed: bool


@dataclass
class GetFeatureVectorCmd:
    addr: int
    offset: int
    size: int
    drop_output_embed: bool


CommandPayload = Union[
    AllocateKvBlocksCmd,
    FillKvBlockCmd,
    FreeKvBlocksCmd,
    AvailableKvBlocksCmd,
    CopyKvBlockCmd,
    MaskKvBlockCmd,
    EmbedImageCmd,
    EmbedVideoCmd
]


@dataclass
class Request:
    instance_id: bytes
    kind: CommandKind
    payload: CommandPayload


# For responses, likewise we define the kind:

class ResponseKind(StrEnum):
    ALLOCATED_BLOCKS = "AllocatedBlocks"
    AVAILABLE_COUNT = "AvailableCount"
    ERROR = "Error"
    AWK = "Awk"


@dataclass
class AllocatedBlocksResp:
    block_id_offset: int
    count: int


@dataclass
class AvailableCountResp:
    count: int


@dataclass
class ErrorResp:
    error_code: int
    message: str


@dataclass
class AwkResp:
    message: str


ResponsePayload = Union[
    AllocatedBlocksResp,
    AvailableCountResp,
    ErrorResp,
    AwkResp
]


@dataclass
class Response:
    instance_id: bytes
    kind: ResponseKind
    payload: ResponsePayload


# Any reusable states, blocks and embeddings
class Resource:
    name: str

    # ones that are not in the list are denied
    access_control: list[str]

    def __init__(self, name: str):
        self.name = name
        self.access_control = []


class ServerState:
    # resource name resolution.

    # public resources should be visible to the users.
    resources: dict[str, str]

    # state management
    block_manager: KvBlockManager
    input_manager: EmbeddingManager
    output_manager: EmbeddingManager

    # command batcher
    fill_cmd_batcher: FillBlockCmdBatcher
    img_cmd_batcher: CreateImageTokensCmdBatcher

    def __init__(self, block_storage: KvBlockStorage, embedding_storage: EmbeddingStorage):
        self.block_manager = KvBlockManager(block_storage)

        # input and output managers actually share the same physical storage
        self.input_manager = EmbeddingManager(embedding_storage)
        self.output_manager = EmbeddingManager(embedding_storage)

        # command batchers
        self.fill_cmd_batcher = FillBlockCmdBatcher()
        self.img_cmd_batcher = CreateImageTokensCmdBatcher()

    def allocate_blocks(self, inst_id: InstanceId, num_blocks: int) -> list[Address]:
        new_block_ids = self.block_manager.allocate_blocks(inst_id, num_blocks)
        return new_block_ids

    def fill_blocks(self, inst_id: InstanceId, block_id: Address, ctx_block_ids: list[Address], block_mask: list[list[bool]], embeddings: list[Token], retain_output_embed: bool):
        # first validate if all the blocks are in the storage

        addr_space = self.block_manager.addr_space[inst_id]

        # translate all block ids into block ptr
        ctx_block_ptrs = []
        for b_id in ctx_block_ids:
            b_ptr = addr_space.resolve(b_id)
            ctx_block_ptrs.append(b_ptr)

        block_ptr = addr_space.resolve(block_id)

        # self.cmd_batcher << accepts block pointers.
        self.fill_cmd_batcher.add
        ...

    def free_blocks(self, inst_id: InstanceId, offset: Address, count: int):
        self.block_manager.delete_blocks(inst_id, list(range(offset, offset + count)))

    def available_blocks(self) -> int:
        return self.block_manager.num_free_blocks()

    def copy_tokens(self, inst_id: InstanceId, src_block_id: Address, dst_block_id: Address, src_offset: int, dst_offset: int, size: int):
        self.block_manager.copy_tokens(inst_id, src_block_id, dst_block_id, src_offset, dst_offset, size)

    def drop_tokens(self, inst_id: InstanceId, block_id: Address, offset: int, size: int):
        self.block_manager.drop_tokens(inst_id, block_id, offset, size)


def parse_incoming_message(msg: list) -> Request:
    """
    The raw message is a list of length 2:
      [ <16-byte UUID>, { "AllocateBlocks": [5] } ]
    or similar for other commands.

    We'll parse them using structural pattern matching.
    """
    if not isinstance(msg, list) or len(msg) != 2:
        raise ValueError("Message must be [uuid_bytes, command_dict]")

    instance_id = msg[0]
    command_dict = msg[1]

    if not isinstance(command_dict, dict):
        (cmd_str, parameters) = (command_dict, [])

    else:
        # There's only one key in command_dict, e.g. {"AllocateBlocks": [5]}
        (cmd_str, parameters) = next(iter(command_dict.items()))

    # Convert string -> CommandKind
    try:
        kind = CommandKind(cmd_str)
    except ValueError:
        raise ValueError(f"Unknown command type: {cmd_str}")

    # In a typical scenario, parameters is a list. We'll match on (kind, parameters).
    match kind, parameters:
        case (CommandKind.ALLOCATE_KV_BLOCKS, [num_blocks]):
            payload = AllocateKvBlocksCmd(num_blocks=num_blocks)

        case (CommandKind.DEALLOCATE_KV_BLOCKS, [offset, count]):
            payload = FreeKvBlocksCmd(addr_offset=offset, count=count)

        case (CommandKind.FILL_KV_BLOCK, [block_id, ctx_block_ids, block_mask, embeddings, retain_output_embed]):
            payload = FillKvBlockCmd(addr=block_id, ctx_addrs=ctx_block_ids, mask=block_mask, tokens=embeddings, retain_output_embed=retain_output_embed)

        case (CommandKind.AVAILABLE_KV_BLOCKS, []):
            payload = AvailableKvBlocksCmd()

        case (CommandKind.COPY_TOKENS, [src, dst, src_offset, dst_offset, size]):
            payload = CopyKvBlockCmd(
                src_addr=src, dst_addr=dst,
                src_offset=src_offset, dst_offset=dst_offset, size=size
            )

        case (CommandKind.MASK_KV_BLOCK, [block_id, offset, size]):
            payload = MaskKvBlockCmd(addr=block_id, offset=offset, size=size)

        case (CommandKind.EMBED_IMAGE, [image_url]):
            payload = EmbedImageCmd(image_url=image_url)

        case (CommandKind.EMBED_VIDEO, [video_url]):
            payload = EmbedVideoCmd(video_url=video_url)

        case (CommandKind.GET_NEXT_TOKEN_DIST, [block_id, offset, size, drop_output_embed]):
            payload = GetNextTokenDistCmd(addr=block_id, offset=offset, size=size, drop_output_embed=drop_output_embed)

        case (CommandKind.GET_FEATURE_VECTOR, [block_id, offset, size, drop_output_embed]):
            payload = GetFeatureVectorCmd(addr=block_id, offset=offset, size=size, drop_output_embed=drop_output_embed)

        case _:
            raise ValueError(f"Invalid parameters for command: {cmd_str}")

    return Request(
        instance_id=instance_id,
        kind=kind,
        payload=payload,
    )


def pad_list(x, max_len, pad_value):
    if len(x) > max_len:
        raise ValueError("List is too long")
    else:
        return x + [pad_value] * (max_len - len(x))


def handle_command(state: ServerState, req: Request) -> tuple[Response | None, bool]:
    """
    Process a typed request, returning a Response if the command needs one.
    """

    # this function may DENY the request if the command is not ready to be processed.

    match req.kind, req.payload:
        case (CommandKind.ALLOCATE_KV_BLOCKS, AllocateKvBlocksCmd(num_blocks)):
            allocated_ids = state.allocate_blocks(num_blocks)
            if not allocated_ids:
                raise ValueError("Allocation returned empty")
            offset = allocated_ids[0]
            count = len(allocated_ids)
            return Response(
                instance_id=req.instance_id,
                kind=ResponseKind.ALLOCATED_BLOCKS,
                payload=AllocatedBlocksResp(block_id_offset=offset, count=count)
            ), True

        case (CommandKind.FILL_KV_BLOCK, FillKvBlockCmd(addr=block_id, ctx_addrs=ctx_block_ids, mask=block_mask, tokens=embeddings, retain_output_embed=retain_output_embed)):

            # inspect the embedding to see if they are ready. if not, return None, False
            for b in embeddings:
                if isinstance(b, ImageToken):
                    if not state.img_cmd_batcher.is_ready(b.image_url):
                        return None, False

            # fill the block with the position id and occupancy mask
            tgt_block = state.block_manager.get_block(req.instance_id, block_id)
            tgt_block.set_occupancy(pad_list([True] * len(embeddings), tgt_block.size, False))
            tgt_block.set_position_ids(pad_list([b.position_id for b in embeddings], tgt_block.size, 0))

            tgt_block.set_filled(False)

            cmd = _FillBlockCmd(
                block=tgt_block,
                ctx_blocks=[state.block_manager.get_block(req.instance_id, b_id) for b_id in ctx_block_ids],
                block_mask=block_mask,
                embeddings=embeddings,
                retain_output_embed=retain_output_embed
            )

            state.fill_cmd_batcher.add(cmd)

            return None, True

        case (CommandKind.DEALLOCATE_KV_BLOCKS, FreeKvBlocksCmd(addr_offset=offset, count=c)):
            state.free_blocks(req.instance_id, offset, c)
            return None, True

        case (CommandKind.AVAILABLE_KV_BLOCKS, AvailableKvBlocksCmd()):
            available = state.available_blocks()
            return Response(
                instance_id=req.instance_id,
                kind=ResponseKind.AVAILABLE_COUNT,
                payload=AvailableCountResp(count=available)
            ), True

        case (CommandKind.COPY_TOKENS, CopyKvBlockCmd(src_addr=src, dst_addr=dst, src_offset=src_offset, dst_offset=dst_offset, size=size)):
            state.copy_tokens(req.instance_id, src, dst, src_offset, dst_offset, size)
            return None, True

        case (CommandKind.MASK_KV_BLOCK, MaskKvBlockCmd(addr=b, offset=offset, size=size)):
            state.drop_tokens(req.instance_id, b, offset, size)
            return None, True

        case (CommandKind.EMBED_IMAGE, EmbedImageCmd(image_url=url)):
            return None, True

        case (CommandKind.EMBED_VIDEO, EmbedVideoCmd(video_url=url)):
            return None, True

        case (CommandKind.GET_NEXT_TOKEN_DIST, GetNextTokenDistCmd(addr=b, offset=offset, size=size, drop_output_embed=drop_output_embed)):
            return None, True

        case (CommandKind.GET_FEATURE_VECTOR, GetFeatureVectorCmd(addr=b, offset=offset, size=size, drop_output_embed=drop_output_embed)):
            return None, True

        case _:
            # This should never happen if all commands are covered
            raise ValueError("Unhandled command variant")


def ceil_div(a, b):
    return -(a // -b)


########
# 1  2  3  4  [13]
# 5  6  7  8  [13]
# 9  10 11 12 [13]
# 13  .  .  . [13]
# 1  2  3  4  [14]
# 5  6  7  8  [14]
# 9  10 11 12 [14]
# 13 14  .  . [14]


# ([Block] [Block] [Block]) -> Chunk
# ([Chunk]
#   [Chunk]
#   [Chunk]) -> Command


#######


class BatchItem:
    target_block_ids: int
    # len(block_ids) = segment_size
    ctx_seg_block_ids: list[int]
    mask_blocks: list[bool]  # indices of the blocks to mask

    def __init__(self, target_block_ids, ctx_seg_block_ids, mask_blocks):
        self.target_block_ids = target_block_ids
        self.ctx_seg_block_ids = ctx_seg_block_ids
        self.mask_blocks = mask_blocks


class ChunkedContext:
    block: KvBlock
    ctx_blocks: list[KvBlock]
    ctx_block_mask: list[bool]


@dataclass
class _FillBlockCmd:
    block: KvBlock
    ctx_blocks: list[KvBlock]
    block_mask: list[bool]
    embeddings: list[Token]
    retain_output_embed: bool


class FillBlockCmdBatcher:
    queue: list[_FillBlockCmd]
    redundancy_check: dict[Address, _FillBlockCmd]

    def __init__(self):
        self.queue = []
        self.redundancy_check = {}

    def add(self, cmd: _FillBlockCmd):

        if cmd.block.pointer not in self.redundancy_check:
            self.redundancy_check[cmd.block.pointer] = cmd
            self.queue.append(cmd)

        else:
            # override the previous command
            prev_cmd = self.redundancy_check[cmd.block.pointer]
            self.queue.remove(prev_cmd)
            self.queue.append(cmd)

    def get_chunk_size(self) -> int:

        # analyze the ideal segment size
        # wasted computation due to sparsity vs. extra reduction operation

        # just use the median of the context block sizes

        num_blocks = []

        for cmd in self.queue:
            num_blocks.append(len(cmd.ctx_blocks))

        return int(np.median(num_blocks).item())

    def batch(self):

        chunk_size = self.get_chunk_size()

        cmd_groups = []  # 2d (NUM_CMDS, MAX_NUM_CHUNKS)

        # N = sum(NUM_CHUNKS per command)

        batched_tgt_block_ptrs = []  # 1d (N, 1)
        batched_ctx_block_ptrs = []  # 2d (N, CHUNK_SIZE)

        batched_token_ids = []  # 2d (N, BLOCK_SIZE)
        batched_pos_ids = []  # 2d (N, BLOCK_SIZE)
        batched_mask = []  # 3d (N, BLOCK_SIZE * CHUNK_SIZE, BLOCK_SIZE)

        token_id_map = []

        for cmd in self.queue:

            num_chunks = ceil_div(len(cmd.ctx_blocks), chunk_size)
            cmd_grp = []
            for i in range(num_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, len(cmd.ctx_blocks))

                ctx_blocks = cmd.ctx_blocks[start:end]
                ctx_block_ptrs = [b.pointer for b in ctx_blocks]
                ctx_block_mask = [False] * chunk_size

                # pad the chunk if it's not full
                if len(ctx_blocks) < chunk_size:
                    # 0 is a special block id for padding
                    pad_size = chunk_size - len(ctx_blocks)
                    ctx_blocks += [0] * pad_size
                    ctx_block_mask[chunk_size - pad_size:] = [True] * pad_size

                # check if the mask is needed
                for mi in cmd.block_mask:
                    if start <= mi < end:
                        ctx_block_mask[mi - start] = True

                # if the entire chunk is masked, we can save some computation. skip it
                if all(ctx_block_mask):
                    continue

                if len(cmd.embeddings) > cmd.block.size:
                    raise ValueError("Too many tokens in the block")

                # get token ids and position ids
                token_ids = []

                for b in cmd.embeddings:
                    if isinstance(b, TextToken):
                        token_ids.append(b.token_id)

                    else:
                        # just fill in stubs. It will be filled in later
                        token_ids.append(0)

                        # get the image token
                        token_id_map.append({
                            "n": len(batched_token_ids),
                            "i": len(token_ids) - 1,
                            "token": b
                        })

                # this is better than computing masks for the entire command.
                # most of the masks will be zeros anyway. so this kinda leverages the sparsity.

                ctx_pos_ids = np.hstack([b.position_ids for b in ctx_blocks])  # int
                tgt_pos_ids = np.array(cmd.block.position_ids)
                ctx_occupancy = np.hstack([b.occupancy for b in ctx_blocks])  # bool

                # get the full attn mask
                block_mask = np.repeat(ctx_block_mask, cmd.block.size)
                casual_mask = ctx_pos_ids[None, :] > tgt_pos_ids[: None]
                valid_mask = np.logical_not(ctx_occupancy[None, :])

                attn_mask = np.logical_or(np.logical_or(casual_mask, valid_mask), block_mask)

                # add the chunk to the batch
                batched_tgt_block_ptrs.append(cmd.block.pointer)
                batched_ctx_block_ptrs.append(ctx_block_ptrs)
                batched_token_ids.append(np.array(token_ids))
                batched_pos_ids.append(tgt_pos_ids)
                batched_mask.append(attn_mask)
                cmd_grp.append(len(batched_token_ids) - 1)

            cmd_groups.append(cmd_grp)

        # create a torch tensor
        batched_tgt_block_ptrs = torch.as_tensor(batched_tgt_block_ptrs)
        batched_ctx_block_ptrs = torch.as_tensor(batched_ctx_block_ptrs)
        batched_token_ids = torch.as_tensor(batched_token_ids)
        batched_pos_ids = torch.as_tensor(batched_pos_ids)
        batched_mask = torch.as_tensor(batched_mask)

        cmd_grp_max_size = max(len(g) for g in cmd_groups)
        cmd_groups = [g + [-1] * (cmd_grp_max_size - len(g)) for g in cmd_groups]
        cmd_groups = torch.as_tensor(cmd_groups)

        return {
            "tgt_block_ptrs": batched_tgt_block_ptrs,
            "ctx_block_ptrs": batched_ctx_block_ptrs,
            "token_ids": batched_token_ids,
            "pos_ids": batched_pos_ids,
            "mask": batched_mask,
            "cmd_groups": cmd_groups,
            "token_id_map": token_id_map
        }

    def clear(self):
        self.queue.clear()
        self.redundancy_check.clear()


class CreateImageTokensCmdBatcher:
    pass
