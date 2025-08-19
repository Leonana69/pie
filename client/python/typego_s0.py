import asyncio
import time
from pathlib import Path
from blake3 import blake3
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

from pie import PieClient, Instance  # Assuming these are defined elsewhere

PREFIX = ""

# ------------------------------
# PIE Runner
# ------------------------------
async def run_inferlet(prompt: str, max_tokens: int, server_uri: str, verbose: bool = False):
    global PREFIX
    program_name = "typego_s0"  # fixed program name
    program_path = Path(f"../../example-apps/target/wasm32-wasip2/release/{program_name}.wasm")

    if not program_path.exists():
        raise FileNotFoundError(f"Program file not found at path: {program_path}")

    async with PieClient(server_uri) as client:
        with open(program_path, "rb") as f:
            program_bytes = f.read()
        program_hash = blake3(program_bytes).hexdigest()

        if not await client.program_exists(program_hash):
            if verbose:
                print("Program not found on server, uploading...")
            await client.upload_program(program_bytes)

        instance_args = [
            "--prompt", prompt,
            "--max-tokens", str(max_tokens),
        ]

        if PREFIX:  # only add if not empty string
            instance_args.extend(["--prefix", PREFIX])

        print(f'Prefix: {PREFIX}')

        instance = await client.launch_instance(program_hash, arguments=instance_args)

        output_messages = []
        while True:
            event, message = await instance.recv()
            if event == "terminated":
                if verbose:
                    print(f"Instance {instance.instance_id} finished. Reason: {message}")
                break
            else:
                output_messages.append(message)
                if verbose:
                    print(f"Instance {instance.instance_id} received message '{message}'")

        print(f"Instance {instance.instance_id} output: {output_messages}")
        return {"prompt": prompt, "output": output_messages}


# ------------------------------
# HTTP Service
# ------------------------------
app = FastAPI()

@app.post("/generate")
async def generate(request: Request):
    global PREFIX
    data = await request.json()
    prompt = data.get("prompt", "Tell me about the number")
    max_tokens = data.get("max_tokens", 64)
    verbose = data.get("verbose", False)
    server_uri = data.get("server_uri", "ws://127.0.0.1:8080")
    prefix = data.get("prefix", "")

    if prefix:
        PREFIX = prefix

    try:
        result = await run_inferlet(prompt, max_tokens, server_uri, verbose)
        return JSONResponse(content={"success": True, "result": result})
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

# ------------------------------
# Entrypoint
# ------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info")
