import asyncio
import msgpack
import websockets
import blake3
import subprocess
import tempfile
from pathlib import Path
import uuid
from enum import Enum


class Event(Enum):
    """Enumeration for events received from an instance."""
    Message = 0
    Completed = 1
    Aborted = 2
    Exception = 3
    ServerError = 4
    OutOfResources = 5
    Blob = 6  # Represents a binary data blob


class Instance:
    """Represents a running instance of a program on the server."""

    def __init__(self, client, instance_id: str):
        self.client = client
        self.instance_id = instance_id
        self.event_queue = self.client.inst_event_queues.get(instance_id)
        if self.event_queue is None:
            raise Exception(f"Internal error: No event queue for instance {instance_id}")

    async def send(self, message: str):
        """Send a string message to the instance."""
        await self.client.signal_instance(self.instance_id, message)

    async def upload_blob(self, blob_bytes: bytes):
        """Upload a blob of binary data to the instance."""
        await self.client.upload_blob(self.instance_id, blob_bytes)

    async def recv(self) -> tuple[Event, str | bytes]:
        """
        Receive an event from the instance. Blocks until an event is available.
        Returns a tuple of (Event, message), where message can be a string or bytes.
        """
        if self.event_queue is None:
            raise Exception("Event queue is not available for this instance.")
        event_code, msg = await self.event_queue.get()

        event = Event(event_code)
        return event, msg

    async def terminate(self):
        """Request termination of the instance."""
        await self.client.terminate_instance(self.instance_id)


class PieClient:
    """
    An asynchronous client for interacting with the Pie WebSocket server.
    This client is designed to be used as an async context manager.
    """

    def __init__(self, server_uri: str):
        """
        Initialize the client.
        :param server_uri: The WebSocket server URI (e.g., "ws://127.0.0.1:8080").
        """
        self.server_uri = server_uri
        self.ws = None
        self.listener_task = None
        self.corr_id_counter = 0
        self.pending_requests = {}
        self.inst_event_queues = {}
        self.pending_downloads = {}  # For reassembling blob chunks

        # 🔻 FIX 1/3: Buffer for early events to prevent race conditions.
        self.orphan_events = {}

    async def __aenter__(self):
        """Enter the async context, establishing the connection."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context, closing the connection cleanly."""
        await self.close()

    async def connect(self):
        """Establish a WebSocket connection and start the background listener."""
        self.ws = await websockets.connect(self.server_uri)
        print(f"[PieClient] Connected to {self.server_uri}")
        self.listener_task = asyncio.create_task(self._listen_to_server())

    async def _listen_to_server(self):
        """Background task to receive and process all incoming server messages."""
        try:
            async for raw_msg in self.ws:
                if isinstance(raw_msg, bytes):
                    try:
                        message = msgpack.unpackb(raw_msg, raw=False)
                        await self._process_server_message(message)
                    except msgpack.UnpackException as e:
                        print(f"[PieClient] Failed to decode messagepack: {e}")
                else:
                    print(f"[PieClient] Received unexpected non-binary message: {raw_msg}")
        except websockets.ConnectionClosedOK:
            print("[PieClient] Connection closed normally.")
        except websockets.ConnectionClosedError as e:
            print(f"[PieClient] Connection closed with error: {e}")
        except Exception as e:
            print(f"[PieClient] Listener task encountered an unexpected error: {e}")

    async def _process_server_message(self, message: dict):
        """Route incoming server messages based on their type."""
        msg_type = message.get("type")
        if msg_type == "response":
            corr_id = message.get("corr_id")
            if corr_id in self.pending_requests:
                future = self.pending_requests.pop(corr_id)
                future.set_result((message.get("successful"), message.get("result")))

        elif msg_type == "instance_event":
            # 🔻 FIX 2/3: Buffer orphan events instead of discarding them.
            instance_id = message.get("instance_id")
            event_tuple = (message.get("event"), message.get("message"))

            if instance_id in self.inst_event_queues:
                # Queue exists, proceed as normal
                await self.inst_event_queues[instance_id].put(event_tuple)
            else:
                # Queue doesn't exist yet, buffer the event
                if instance_id not in self.orphan_events:
                    self.orphan_events[instance_id] = []
                self.orphan_events[instance_id].append(event_tuple)

        elif msg_type == "download_blob":
            await self._handle_blob_chunk(message)
        elif msg_type == "server_event":
            print(f"[PieClient] Received server event: {message.get('message')}")
        else:
            print(f"[PieClient] Received unknown message type: {msg_type}")

    async def _handle_blob_chunk(self, message: dict):
        """Processes a chunk of a blob sent from the server, ensuring sequential order."""
        blob_hash = message.get("blob_hash")
        instance_id = message.get("instance_id")
        chunk_index = message.get("chunk_index")
        total_chunks = message.get("total_chunks")

        if instance_id not in self.inst_event_queues:
            return  # Ignore blobs for unknown/terminated instances

        # Initialize download on the first chunk (index 0)
        if blob_hash not in self.pending_downloads:
            if chunk_index != 0:
                print(f"[PieClient] Received non-zero first chunk for blob {blob_hash}. Discarding.")
                return
            self.pending_downloads[blob_hash] = {
                "buffer": bytearray(),
                "total_chunks": total_chunks,
                "next_chunk_index": 1,
                "instance_id": instance_id,
            }

        download = self.pending_downloads[blob_hash]

        # Validate chunk consistency and order
        if total_chunks != download["total_chunks"] or chunk_index != download["next_chunk_index"] - 1:
            error_msg = "Chunk count mismatch" if total_chunks != download["total_chunks"] else "Out-of-order chunk"
            print(f"[PieClient] {error_msg} for blob {blob_hash}. Aborting download.")
            del self.pending_downloads[blob_hash]
            return

        download["buffer"].extend(message.get("chunk_data"))
        download["next_chunk_index"] += 1

        # If all chunks are received, finalize the download
        if download["next_chunk_index"] == download["total_chunks"]:
            completed_blob = bytes(download["buffer"])
            computed_hash = blake3.blake3(completed_blob).hexdigest()
            if computed_hash == blob_hash:
                await self.inst_event_queues[instance_id].put((Event.Blob.value, completed_blob))
            else:
                print(f"[PieClient] Blob hash mismatch for instance {instance_id}. Expected {blob_hash}, got {computed_hash}. Discarding.")

            del self.pending_downloads[blob_hash]

    async def close(self):
        """Gracefully close the WebSocket connection and shut down background tasks."""
        if self.ws and not self.ws.closed:
            await self.ws.close()
        if self.listener_task:
            try:
                self.listener_task.cancel()
                await self.listener_task
            except asyncio.CancelledError:
                pass  # Expected on cancellation
        print("[PieClient] Client has been shut down.")

    def _get_next_corr_id(self):
        """Generate a unique correlation ID for a request."""
        self.corr_id_counter += 1
        return self.corr_id_counter

    async def _send_msg_and_wait(self, msg: dict) -> tuple[bool, str]:
        """Send a message that expects a response and wait for it."""
        corr_id = self._get_next_corr_id()
        msg["corr_id"] = corr_id
        future = asyncio.get_event_loop().create_future()
        self.pending_requests[corr_id] = future
        encoded = msgpack.packb(msg, use_bin_type=True)
        await self.ws.send(encoded)
        return await future

    async def authenticate(self, token: str) -> tuple[bool, str]:
        """Authenticate the client with the server using a token."""
        msg = {"type": "authenticate", "token": token}
        successful, result = await self._send_msg_and_wait(msg)
        if successful:
            print("[PieClient] Authenticated successfully.")
        else:
            print(f"[PieClient] Authentication failed: {result}")
        return successful, result

    async def query(self, subject: str, record: str) -> tuple[bool, str]:
        """Send a generic query to the server."""
        msg = {"type": "query", "subject": subject, "record": record}
        return await self._send_msg_and_wait(msg)

    async def program_exists(self, program_hash: str) -> bool:
        """Check if a program with the given hash exists on the server."""
        successful, result = await self.query("program_exists", program_hash)
        if successful:
            return result == "true"
        raise Exception(f"Query for program_exists failed: {result}")

    async def _upload_chunked(self, data_bytes: bytes, msg_template: dict):
        """Internal helper to handle generic chunked uploads."""
        data_hash = msg_template.get("program_hash") or msg_template.get("blob_hash")
        upload_type = msg_template["type"]

        chunk_size = 256 * 1024
        total_size = len(data_bytes)
        # An empty upload is still one chunk of zero bytes
        total_chunks = (total_size + chunk_size - 1) // chunk_size if total_size > 0 else 1

        corr_id = self._get_next_corr_id()
        msg_template["corr_id"] = corr_id
        msg_template["total_chunks"] = total_chunks

        if total_size == 0:
            msg = msg_template.copy()
            msg.update({"chunk_index": 0, "chunk_data": b''})
            await self.ws.send(msgpack.packb(msg, use_bin_type=True))
        else:
            for chunk_index in range(total_chunks):
                start = chunk_index * chunk_size
                end = min(start + chunk_size, total_size)
                msg = msg_template.copy()
                msg.update({"chunk_index": chunk_index, "chunk_data": data_bytes[start:end]})
                await self.ws.send(msgpack.packb(msg, use_bin_type=True))

        future = asyncio.get_event_loop().create_future()
        self.pending_requests[corr_id] = future
        successful, result = await future

        if not successful:
            raise Exception(f"{upload_type.replace('_', ' ').title()} failed: {result}")

        print(f"[PieClient] {upload_type.replace('_', ' ').title()} successful for hash: {data_hash}")
        return result

    async def upload_program(self, program_bytes: bytes):
        """Upload a program to the server in chunks."""
        program_hash = blake3.blake3(program_bytes).hexdigest()
        template = {"type": "upload_program", "program_hash": program_hash}
        await self._upload_chunked(program_bytes, template)

    async def upload_blob(self, instance_id: str, blob_bytes: bytes):
        """Upload a blob of data to a specific instance in chunks."""
        blob_hash = blake3.blake3(blob_bytes).hexdigest()
        template = {"type": "upload_blob", "instance_id": instance_id, "blob_hash": blob_hash}
        await self._upload_chunked(blob_bytes, template)

    async def launch_instance(self, program_hash: str, arguments: list[str] = None) -> Instance:
        """Launch an instance of a program."""
        msg = {"type": "launch_instance", "program_hash": program_hash, "arguments": arguments or []}
        successful, result = await self._send_msg_and_wait(msg)
        if successful:
            instance_id = result
            # Create the queue as before
            queue = asyncio.Queue()
            self.inst_event_queues[instance_id] = queue

            # 🔻 FIX 3/3: Check for and replay any events that arrived early.
            if instance_id in self.orphan_events:
                early_events = self.orphan_events.pop(instance_id)
                for event_tuple in early_events:
                    await queue.put(event_tuple)

            return Instance(self, instance_id)
        raise Exception(f"Failed to launch instance: {result}")

    async def launch_server_instance(self, program_hash: str, port: int, arguments: list[str] = None):
        """Launch a server instance of a program on a specific port."""
        msg = {"type": "launch_server_instance", "port": port, "program_hash": program_hash, "arguments": arguments or []}
        successful, result = await self._send_msg_and_wait(msg)
        if not successful:
            raise Exception(f"Failed to launch server instance: {result}")

    async def signal_instance(self, instance_id: str, message: str):
        """Send a signal/message to a running instance (fire-and-forget)."""
        msg = {"type": "signal_instance", "instance_id": instance_id, "message": message}
        await self.ws.send(msgpack.packb(msg, use_bin_type=True))

    async def terminate_instance(self, instance_id: str):
        """Request the server to terminate a running instance (fire-and-forget)."""
        msg = {"type": "terminate_instance", "instance_id": instance_id}
        await self.ws.send(msgpack.packb(msg, use_bin_type=True))


def _compile_rust_sync(rust_code: str, cargo_toml_content: str, package_name: str) -> bytes:
    """[Internal Synchronous Helper] Compiles rust code in a temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        (project_path / "src").mkdir()
        (project_path / "Cargo.toml").write_text(cargo_toml_content)
        (project_path / "src" / "lib.rs").write_text(rust_code)
        command = ["cargo", "build", "--target", "wasm32-wasip2", "--release"]
        try:
            print(f"🚀 Compiling crate '{package_name}'...")
            subprocess.run(command, cwd=project_path, check=True, capture_output=True, text=True)
        except FileNotFoundError:
            raise RuntimeError("Error: `cargo` not found. Is Rust installed? Try: `rustup target add wasm32-wasip2`")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"❌ Rust compilation failed.\n--- COMPILER OUTPUT ---\n{e.stderr}")
        wasm_file_name = f"{package_name.replace('-', '_')}.wasm"
        wasm_path = project_path / "target" / "wasm32-wasip2" / "release" / wasm_file_name
        if not wasm_path.exists():
            raise RuntimeError(f"Build succeeded but could not find WASM file at {wasm_path}")
        print("✅ Compilation successful! Reading WASM binary.")
        return wasm_path.read_bytes()


async def compile_program(source: str | Path, dependencies: list[str]) -> bytes:
    """Compiles Rust source into a WASM binary and returns the bytes."""
    if isinstance(source, Path) or (isinstance(source, str) and source.endswith('.rs')):
        rust_code = Path(source).read_text()
    else:
        rust_code = source
    package_name = f"pie-temp-crate-{uuid.uuid4().hex[:8]}"
    deps_str = "\n".join(dependencies)
    cargo_toml_content = f"""
[package]
name = "{package_name}"
version = "0.1.0"
edition = "2021"
[lib]
crate-type = ["cdylib"]
[dependencies]
{deps_str}
"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _compile_rust_sync, rust_code, cargo_toml_content, package_name)