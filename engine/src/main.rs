// #![allow(unused)]
//
mod client;
mod service;
mod zmq_handler;

mod backend;
mod batching;
mod bindings;
mod instance;
mod l4m;
mod messaging;
mod object;
mod ping;
mod runtime;
mod server;
mod tokenizer;
mod utils;

//
use anyhow::Context;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

use crate::client::{Client, hash_program};
use crate::l4m::L4m;
use crate::messaging::{PubSub, PushPull};
use crate::ping::Ping;
use crate::runtime::Runtime;
use crate::zmq_handler::{ManagementConfig, check_management_service_status, get_model_endpoint};
use crate::server::Server;
use crate::service::Controller;
use clap::{Arg, Command};
use colored::Colorize;
use std::fs;

const PROGRAM_CACHE_DIR: &str = "./program_cache";

//
// Engine configuration structures - compatible with unified config
#[derive(Serialize, Deserialize, Debug)]
struct Config {
    system: SystemConfig,
    services: ServicesConfig,
    endpoints: EndpointsConfig,
    logging: LoggingConfig,
    models: ModelsConfig,
    backends: BackendsConfig,
    paths: PathsConfig,
}

#[derive(Serialize, Deserialize, Debug)]
struct SystemConfig {
    name: String,
    version: String,
    description: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct ServicesConfig {
    engine_manager: EngineManagerConfig,
    engine: EngineConfig,
}

#[derive(Serialize, Deserialize, Debug)]
struct EngineManagerConfig {
    host: String,
    port: u16,
    binary_name: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct EngineConfig {
    binary_name: String,
    default_port: u16,
    base_args: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct EndpointsConfig {
    client_handshake: String,
    cli_management: String,
    management_service: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct LoggingConfig {
    level: String,
    format: String,
    date_format: String,
    directory: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct ModelsConfig {
    available: Vec<String>,
    default: String,
    supported_models: Vec<SupportedModel>,
}

#[derive(Serialize, Deserialize, Debug)]
struct SupportedModel {
    name: String,
    fullname: String,
    #[serde(rename = "type")]
    model_type: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct BackendsConfig {
    model_backends: std::collections::HashMap<String, String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct PathsConfig {
    engine_binary_search: Vec<String>,
    engine_manager_binary_search: Vec<String>,
}

// Legacy EngineConfig structure for backward compatibility during transition
#[derive(Serialize, Deserialize, Debug)]
struct LegacyEngineConfig {
    management_service: ManagementServiceConfig,
    models: Vec<String>,
    default_model: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct ManagementServiceConfig {
    endpoint: String,
}

//
// Define a simple macro for client-side logging.
#[macro_export]
macro_rules! log_user {
    ($($arg:tt)*) => {
        println!("{}", format!("[User] {}", format_args!($($arg)*)).bright_blue());
    }
}

//use console_subscriber;

/// Load engine configuration from JSON file
fn load_config(config_path: Option<&str>) -> anyhow::Result<(Vec<String>, String, String)> {
    let config_path = config_path.unwrap_or("./config.json");

    let config_content = std::fs::read_to_string(config_path)
        .with_context(|| format!("Failed to read config file: {}", config_path))?;

    // Try to parse as unified config first
    if let Ok(config) = serde_json::from_str::<Config>(&config_content) {
        log_user!("Loaded unified config from: {}", config_path);

        let models = config.models.available;
        let default_model = config.models.default;
        let management_endpoint = config.endpoints.management_service;

        return Ok((models, default_model, management_endpoint));
    }

    // Fall back to legacy config format
    if let Ok(legacy_config) = serde_json::from_str::<LegacyEngineConfig>(&config_content) {
        log_user!("Loaded legacy config from: {}", config_path);

        let models = legacy_config.models;
        let default_model = legacy_config.default_model;
        let management_endpoint = legacy_config.management_service.endpoint;

        return Ok((models, default_model, management_endpoint));
    }

    Err(anyhow::anyhow!("Failed to parse config file as either unified or legacy format: {}", config_path))
}

/// Check all models in config and return the first available one
async fn find_first_available_model(models: &[String], management_endpoint: &str) -> anyhow::Result<String> {
    let mgmt_config = ManagementConfig {
        endpoint: management_endpoint.to_string(),
    };

    for model_name in models {
        // Try to get the model endpoint, which will load the model if it's not already loaded
        match get_model_endpoint(model_name, &mgmt_config).await {
            Ok(_endpoint) => {
                return Ok(model_name.clone());
            }
            Err(e) => {
                log_user!("Failed to load model {}: {}", model_name, e);
                continue;
            }
        }
    }

    Err(anyhow::anyhow!("No available models found from the configured list: {:?}", models))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    //console_subscriber::init();

    // Parse command line arguments
    let matches = Command::new("Symphony Engine")
        .version("1.0")
        .author("Symphony Team")
        .about("Symphony Engine for WebAssembly programs")
        .arg(
            Arg::new("program")
                .help("Name of the program to run (without extension)")
                .default_value("simple_decoding"),
        )
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("CONFIG_PATH")
                .help("Path to engine config file")
                .default_value("./config.json"),
        )
        .arg(
            Arg::new("http")
                .short('H')
                .long("http")
                .action(clap::ArgAction::SetTrue)
                .help("Run the HTTP server")
                .default_value("false"),
        )
        .arg(
            Arg::new("port")
                .short('p')
                .long("port")
                .value_name("PORT")
                .help("Port to run the HTTP server on")
                .default_value("9123"),
        )
        .arg(
            Arg::new("dummy")
                .short('d')
                .long("dummy")
                .action(clap::ArgAction::SetTrue)
                .help("Run the dummy client")
                .default_value("false"),
        )
        .get_matches();

    let program_name = matches.get_one::<String>("program").unwrap();
    let config_path = matches.get_one::<String>("config").unwrap();
    let is_http = matches.get_one::<bool>("http").unwrap();
    let port = matches.get_one::<String>("port").unwrap();
    let port: u16 = port.parse().unwrap_or(9123);
    let use_dummy = matches.get_one::<bool>("dummy").unwrap();
    let use_dummy = *use_dummy;

    // Load engine configuration
    let (models, _default_model, management_endpoint) = load_config(Some(config_path))?;

    // Check if management service is running first
    let mgmt_config = ManagementConfig {
        endpoint: management_endpoint.clone(),
    };
    check_management_service_status(&mgmt_config).await
        .context("Management service is not available")?;

    // Find the first available model from the config
    let model_name = find_first_available_model(&models, &management_endpoint).await
        .context("Failed to find any available models")?;

    log_user!("Using model: {}", model_name);

    // 1) Ensure the cache directory exists
    fs::create_dir_all(PROGRAM_CACHE_DIR).context("Failed to create program cache directory")?;

    //return Ok(());

    let runtime = Runtime::new();
    runtime.load_existing_programs(Path::new(PROGRAM_CACHE_DIR))?;

    let server = Server::new("127.0.0.1:9123");
    let messaging_inst2inst = PubSub::new();
    let messaging_user2inst = PushPull::new();


    let ctrl = Controller::new()
            .add("runtime", runtime)
            .add("server", server)
            .add("messaging-inst2inst", messaging_inst2inst)
            .add("messaging-user2inst", messaging_user2inst);

    // Setup with dummy
    let ctrl = if use_dummy {
        log_user!("Running in dummy mode");

        let dummy_l4m_backend = backend::SimulatedBackend::new(l4m::Simulator::new()).await;
        let dummy_ping_backend = backend::SimulatedBackend::new(ping::Simulator::new()).await;

        let l4m = L4m::new(dummy_l4m_backend.clone()).await;
        let ping = Ping::new(dummy_ping_backend.clone()).await;

        ctrl
            .add(l4m::available_models().first().unwrap(), l4m)
            .add("ping", ping)

    } else {
        // Get the model endpoint from management service
        let model_endpoint = get_model_endpoint(&model_name, &mgmt_config).await
            .context("Failed to get model endpoint from management service")?;

        // Connect to the model backend endpoint
        let l4m_backend = backend::ZmqBackend::bind(&model_endpoint).await?;

        // Setup with real backend
        let l4m = L4m::new(l4m_backend.clone()).await;
        let ping = Ping::new(l4m_backend.clone()).await;

        // Install all services
        ctrl
            .add(l4m::available_models().first().unwrap(), l4m)
            .add("ping", ping)
    };

    // Install all services
    ctrl.install();

    // periodically print stats
    tokio::spawn(async {
        loop {
            let service_id = service::get_service_id(l4m::available_models().first().unwrap()).unwrap();
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            l4m::Command::PrintStats.dispatch(service_id).unwrap();
        }
    });

    // TEST: spawn a dummy client with the program name
    if *is_http {
        tokio::spawn(dummy_client2(program_name.to_string(), port));
    } else {
        tokio::spawn(dummy_client(program_name.to_string()));
    }
    // wait forever
    tokio::signal::ctrl_c().await?;

    Ok(())
}

async fn dummy_client2(program_name: String, port: u16) -> anyhow::Result<()> {
    let program_path = PathBuf::from(format!(
        "../example-apps/target/wasm32-wasip2/release/{}.wasm",
        program_name
    ));

    let server_uri = "ws://127.0.0.1:9123";

    let mut client = Client::connect(server_uri).await?;
    let program_blob = fs::read(&program_path)?;
    let program_hash = hash_program(&program_blob);

    log_user!("Program file hash: {}", program_hash);

    // If program is not present, upload it
    if !client.program_exists(&program_hash).await? {
        log_user!("Program not found on server, uploading now...");
        client.upload_program(&program_blob).await?;
        log_user!("Program uploaded successfully!");
    }

    client
        .launch_server_instance(&program_hash, port as u32)
        .await?;

    Ok(())
}

async fn dummy_client(program_name: String) -> anyhow::Result<()> {
    let program_path = PathBuf::from(format!(
        "../example-apps/target/wasm32-wasip2/release/{}.wasm",
        program_name
    ));

    // Check if the file exists
    if !program_path.exists() {
        log_user!("Error: Program file not found at path: {:?}", program_path);
        return Ok(());
    }

    let server_uri = "ws://127.0.0.1:9123";

    log_user!("Using program: {}", program_name);

    let mut client = Client::connect(server_uri).await?;
    let program_blob = fs::read(&program_path)?;
    let program_hash = hash_program(&program_blob);

    log_user!("Program file hash: {}", program_hash);

    // If program is not present, upload it
    if !client.program_exists(&program_hash).await? {
        log_user!("Program not found on server, uploading now...");
        client.upload_program(&program_blob).await?;
        log_user!("Program uploaded successfully!");
    }

    const NUM_INSTANCES: usize = 1;

    // Launch 1 instances sequentially
    let mut instances = Vec::new();
    for i in 0..NUM_INSTANCES {
        let instance = client.launch_instance(&program_hash).await?;
        log_user!("Instance {} launched.", instance.id());
        instances.push(instance);
    }

    // Spawn a task for each instance to handle sending and receiving concurrently.
    let mut handles = Vec::new();
    for mut instance in instances {
        let handle = tokio::spawn(async move {
            instance.send("event #1: Hello from Rust client").await?;
            instance.send("event #2: Another event").await?;
            instance.send("event #3: Another event").await?;

            while let Ok((event, message)) = instance.recv().await {
                match event.as_str() {
                    "terminated" => {
                        log_user!("Instance {} terminated. Reason: {}", instance.id(), message);
                        break;
                    }
                    _ => {
                        log_user!("Instance {} received message: {}", instance.id(), message);
                    }
                }
            }
            anyhow::Result::<()>::Ok(())
        });
        handles.push(handle);
    }

    // Wait for all instance tasks to complete.
    for handle in handles {
        handle.await??;
    }

    client.close().await?;
    log_user!("Client connection closed.");
    Ok(())
}
