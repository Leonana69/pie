use std::time::Instant;
use symphony::RunSync;

struct PrefixSharing;

// create a default stream constant

impl RunSync for PrefixSharing {
    fn run() -> Result<(), String> {
        let start = Instant::now();

        let max_num_outputs = 128;

        let mut ctx = symphony::Context::new();
        ctx.fill("<|begin_of_text|>");
        ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");

        let mut ctx_sub1 = ctx.fork();
        let mut ctx_sub2 = ctx.fork();

        ctx_sub1.fill("<|start_header_id|>user<|end_header_id|>\n\nExplain the LLM decoding process ELI5.<|eot_id|>");
        ctx_sub1.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");

        ctx_sub2.fill("<|start_header_id|>user<|end_header_id|>\n\nExplain the Espresso making process ELI5.<|eot_id|>");
        ctx_sub2.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");

        let output_text1 = ctx_sub1.generate("<|eot_id|>", max_num_outputs);

        println!(
            "Output: {:?} (elapsed: {:?})",
            output_text1,
            start.elapsed()
        );

        let output_text2 = ctx_sub2.generate("<|eot_id|>", max_num_outputs);

        println!(
            "Output: {:?} (elapsed: {:?})",
            output_text2,
            start.elapsed()
        );

        Ok(())
    }
}

symphony::main_sync!(PrefixSharing);
