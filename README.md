<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://pie-project.org/images/pie-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://pie-project.org/images/pie-light.svg">
    <img alt="Pie: Programmable serving system for emerging LLM applications"
         src="https://pie-project.org/images/pie-light.svg"
         width="30%">
    <p>
    <strong>High-performance, programmable LLM serving</strong>
  </p>
  </picture>

[Getting started] | [Learn] | [Documentation] | [Contributing]
</div>

This is the main source code repository for [pie], a programmable inference engine for LLM applications

[pie]: https://pie-project.org/
[Getting Started]: https://pie-project.org/learn/get-started
[Learn]: https://pie-project.org/learn
[Documentation]: https://pie-project.org/learn#learn-use
[Contributing]: CONTRIBUTING.md

## Getting Started

**1. Install the WASI Preview 2 Target**
```bash
rustup target add wasm32-wasip2
```

**2. Compile the Example LIP App**  
There are example applications located in `example-apps/`. They are written in Rust which will be compiled to WebAssembly using the `wasm32-wasip2` target.
- To build example applications, from the root of the repository, run:
```bash
cd example-apps
cargo build --target wasm32-wasip2 --release
```
This uses `wit-bindgen` to implement the `spi:app/run` interface. The compiled `helloworld.wasm` file is located in `./example-apps/target/wasm32-wasip2/release/`.

**3. Compile and Run Backend**
Here we use `pytorch` as an example backend.
- From the root of the repository, run:
```bash
cd backend/backend-pytorch
pip install -r requirements.txt
python main.py
```

**4. Compile Symphony Engine**
Now we will compile the Symphony engine. The engine will automatically run `simple-decoding` example application currnetly hard-coded at [here](https://github.com/symphony-project/symphony/blob/d0193f224c0f98a029a3356b2f83344992367740/engine/src/main.rs#L90).

From the root of the repository, run:
```bash
cd engine
./download_tokenizer.sh
cargo build --release
cargo run --release
```
