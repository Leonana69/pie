//! CLI command definitions and parsing logic.

use clap::Parser;
use super::zmq_client;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct CliArgs {
    #[clap(subcommand)]
    pub command: Commands,
    
    /// Output responses in JSON format
    #[clap(long, global = true)]
    pub json: bool,
}

#[derive(Parser, Debug)]
pub enum Commands {
    /// Start the management service (daemon)
    StartService {
        #[clap(long, action)]
        daemonize: bool,
    },
    /// Stop the management service
    StopService,
    /// Get the status of the management service
    Status,
    /// Load a model
    LoadModel {
        model_name: String,
        #[clap(long)]
        config_path: Option<String>,
    },
    /// Unload a model
    UnloadModel {
        model_name: String,
    },
    /// List loaded models
    ListModels,
    /// Install a model from HuggingFace Hub
    InstallModel {
        /// Model name or path on HuggingFace Hub (e.g., meta-llama/Llama-3.1-8B-Instruct)
        model_name: String,
        /// Local name to use for the model (optional, defaults to last part of model_name)
        #[clap(long)]
        local_name: Option<String>,
        /// Force reinstall even if model already exists
        #[clap(long, action)]
        force: bool,
    },
    /// Uninstall a model from local storage
    UninstallModel {
        /// Model name to uninstall
        model_name: String,
        /// Force uninstall even if model is currently loaded
        #[clap(long, action)]
        force: bool,
    },
    // TODO: Add other commands as needed, e.g., health, logs
}

pub async fn process_cli_command(args: CliArgs) {
    match args.command {
        Commands::StartService { daemonize } => {
            handle_start_service(daemonize, args.json).await;
        }
        other_command => {
            // Use ZMQ client for all other commands
            match zmq_client::send_command_to_service(other_command, args.json).await {
                Ok(response) => println!("{}", response),
                Err(e) => {
                    if args.json {
                        println!("{}", serde_json::json!({"error": e}));
                    } else {
                        eprintln!("Error: {}", e);
                    }
                }
            }
        }
    }
}

async fn handle_start_service(daemonize: bool, json: bool) {
    // First check if service is already running
    match zmq_client::send_command_to_service(Commands::Status, json).await {
        Ok(_) => {
            if json {
                println!("{}", serde_json::json!({"status": "already_running", "message": "Service is already running"}));
            } else {
                println!("Service is already running.");
            }
            return;
        }
        Err(_) => {
            // Service is not running, we can try to start it
        }
    }
    
    // Find the symphony-management-service binary
    let binary_path = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|p| p.join("symphony-management-service")))
        .filter(|path| path.exists())
        .unwrap_or_else(|| std::path::PathBuf::from("./target/release/symphony-management-service"));
    
    if !binary_path.exists() {
        let error_msg = format!("Could not find symphony-management-service binary at: {}", binary_path.display());
        if json {
            println!("{}", serde_json::json!({"status": "error", "message": error_msg}));
        } else {
            eprintln!("Error: {}", error_msg);
        }
        return;
    }
    
    // Start the service
    let mut command = std::process::Command::new(&binary_path);
    
    if daemonize {
        // For daemonization, we could use nohup or implement proper daemonization
        // For now, we'll start it in the background
        command.stdout(std::process::Stdio::null())
               .stderr(std::process::Stdio::null())
               .stdin(std::process::Stdio::null());
    }
    
    match command.spawn() {
        Ok(mut child) => {
            if daemonize {
                // For daemonized mode, we don't wait for the child
                if json {
                    println!("{}", serde_json::json!({
                        "status": "started", 
                        "message": "Service started in background",
                        "pid": child.id()
                    }));
                } else {
                    println!("✓ Service started in background (PID: {})", child.id());
                }
            } else {
                if json {
                    println!("{}", serde_json::json!({
                        "status": "starting", 
                        "message": "Service starting...",
                        "pid": child.id()
                    }));
                } else {
                    println!("✓ Service starting... (PID: {})", child.id());
                    println!("Press Ctrl+C to stop the service");
                }
                
                // Wait for the child process
                match child.wait() {
                    Ok(status) => {
                        if json {
                            println!("{}", serde_json::json!({
                                "status": "exited", 
                                "exit_code": status.code()
                            }));
                        } else {
                            println!("Service exited with code: {:?}", status.code());
                        }
                    }
                    Err(e) => {
                        if json {
                            println!("{}", serde_json::json!({"status": "error", "message": format!("Failed to wait for service: {}", e)}));
                        } else {
                            eprintln!("Error waiting for service: {}", e);
                        }
                    }
                }
            }
        }
        Err(e) => {
            let error_msg = format!("Failed to start service: {}", e);
            if json {
                println!("{}", serde_json::json!({"status": "error", "message": error_msg}));
            } else {
                eprintln!("Error: {}", error_msg);
            }
        }
    }
}
