<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://pie-project.org/images/pie-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://pie-project.org/images/pie-light.svg">
    <img alt="Pie: Programmable serving system for emerging LLM applications"
         src="https://pie-project.org/images/pie-light.svg"
         width="30%">
    <p></p>
  </picture>
</div>


**Pie** is a high-performance, programmable LLM serving system that empowers you to design and deploy custom inference logic and optimization strategies.

> **Note** 🧪
>
> This software is in a **pre-release** stage and under active development. It's recommended for testing and research purposes only.



## Getting Started

### 1. Prerequisites

- **Configure a Backend:**  
  Navigate to a backend directory and follow its `README.md` for setup:
    - [Python Backend](backend/backend-python/README.md)


- **Add Wasm Target:**  
  Install the WebAssembly target for Rust:

  ```bash
  rustup target add wasm32-wasip2
  ```
  This is required to compile Rust-based inferlets in the `example-apps` directory.


### 2. Build

Build the **PIE CLI** and the example inferlets.

- **Build the PIE CLI:**  
  From the repository root, run:

  ```bash
  cd pie-cli && cargo install --path .
  ```

- **Build the Examples:**

  ```bash
  cd example-apps && cargo build --target wasm32-wasip2 --release
  ```



### 3. Run an Inferlet

Download a model, start the engine, and run an inferlet.

1. **Download a Model:**  
   Use the PIE CLI to add a model from the [model index](https://github.com/pie-project/model-index):

   ```bash
   pie model add "llama-3.2-1b-instruct"
   ```

2. **Start the Engine:**  
   Launch the PIE engine with an example configuration. This opens the interactive PIE shell:

   ```bash
   cd pie-cli
   pie start --config ./example_config.toml
   ```

3. **Run an Inferlet:**  
   From within the PIE shell, execute a compiled inferlet:

   ```bash
   pie> run ../example-apps/target/wasm32-wasip2/release/text_completion.wasm -- --prompt "What is the capital of France?"
   ```

