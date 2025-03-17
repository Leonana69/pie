pub mod runtime;
use once_cell::sync::OnceCell;

use crate::instance::Id as InstanceId;
use crate::object;
use crate::object::ObjectError;
use crate::runtime::ExceptionDispatcher;
use std::any::{Any, TypeId};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::mem;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::mpsc::{Receiver, Sender, UnboundedSender, channel, unbounded_channel};
use tokio::task;

// Common driver routines
pub static SERVICE_RUNTIME: usize = 0;
pub static SERVICE_SERVER: usize = 1;
pub static SERVICE_MESSAGING: usize = 2;

static SERVICE_DISPATCHER: OnceCell<ServiceDispatcher> = OnceCell::new();

pub fn dispatch<C>(service_idx: usize, cmd: C) -> Result<(), ServiceError>
where
    C: Any + Send + Sync + Debug + 'static,
{
    SERVICE_DISPATCHER
        .get()
        .expect("Service not initialized")
        .dispatch(service_idx, cmd)
}

#[derive(Debug, Error)]
pub enum ServiceError {
    #[error("Driver '{0}' not found")]
    DriverNotFound(String),

    #[error("Invalid driver index: {0}")]
    InvalidDriverIndex(usize),
}

pub trait Service {
    type Command: Send + Sync + 'static;

    // fn create(&mut self, inst: InstanceId) {}
    //
    // fn destroy(&mut self, inst: InstanceId) {}

    async fn handle(&mut self, cmd: Self::Command);

    // fn reporter(&self) -> Option<&ExceptionDispatcher> {
    //     None
    // }
}

pub type AnyCommand = Box<dyn Any + Send + Sync>;

pub struct ServiceInstaller {
    maps: HashMap<String, usize>,
    channels: Vec<UnboundedSender<AnyCommand>>,
}

impl ServiceInstaller {
    pub fn new() -> Self {
        let builder = Self {
            maps: HashMap::new(),
            channels: Vec::new(),
        };
        builder.add_builtin_services()
    }

    fn add_builtin_services(mut self) -> Self {
        self.add("runtime", runtime::RuntimeHelper::new("0.1.0"))
            .add("server", object::Server::new())
            .add("messaging", object::Messaging::new())
    }

    pub fn add<T>(mut self, name: &str, mut driver: T) -> Self
    where
        T: Service + 'static,
    {
        let (tx, mut rx) = unbounded_channel();

        self.channels.push(tx);
        self.maps.insert(name.to_string(), self.channels.len() - 1);

        task::spawn(async move {
            while let Some(cmd) = rx.recv().await {
                driver.handle(cmd.downcast().unwrap()).await
            }
        });

        self
    }

    pub fn install(self) {
        let dispatcher = ServiceDispatcher {
            maps: self.maps,
            channels: self.channels,
        };
        SERVICE_DISPATCHER
            .set(dispatcher)
            .expect("Dispatcher already initialized");
    }
}

#[derive(Debug)]
pub struct ServiceDispatcher {
    maps: HashMap<String, usize>,
    channels: Vec<UnboundedSender<AnyCommand>>,
}

impl ServiceDispatcher {
    pub fn get_service_idx(&self, name: &str) -> Option<usize> {
        self.maps.get(name).map(|idx| *idx)
    }

    pub fn dispatch<C>(&self, service_idx: usize, cmd: C) -> Result<(), ServiceError>
    where
        C: Any + Send + Sync + Debug + 'static,
    {
        let cmd = Box::new(cmd);

        self.channels
            .get(service_idx)
            .ok_or(ServiceError::InvalidDriverIndex(service_idx))?
            .send(cmd)
            .unwrap();

        Ok(())
    }

    pub fn dispatch_with<C>(&self, driver_name: &str, cmd: C) -> Result<(), ServiceError>
    where
        C: Any + Send + Sync + Debug + 'static,
    {
        let driver_idx = self
            .maps
            .get(driver_name)
            .ok_or(ServiceError::DriverNotFound(driver_name.to_string()))?;

        self.dispatch(*driver_idx, cmd)?;

        Ok(())
    }
}

// Some common driver utilities
//
// pub enum Operation<C> {
//     Create(InstanceId),
//     Destroy(InstanceId),
//     Dispatch(InstanceId, C),
// }
//
// // Router -- routes commands to the appropriate driver based on the command type
//
// pub struct AnyCommand {
//     orig_type_id: TypeId,
//     bx: Box<dyn Any + Send + Sync>,
// }
//
// /// -----------
//
// impl AnyCommand {
//     pub fn new<T: Any + Send + Sync>(cmd: T) -> Self {
//         Self {
//             orig_type_id: TypeId::of::<T>(),
//             bx: Box::new(cmd),
//         }
//     }
//
//     pub fn into_inner<T: Any + Send + Sync + Debug>(self) -> Result<T, AnyCommand> {
//         if self.orig_type_id == TypeId::of::<T>() {
//             Ok(*self.bx.downcast().unwrap())
//         } else {
//             Err(self)
//         }
//     }
// }
//
// pub struct Router {
//     channels: HashMap<TypeId, Sender<Operation<AnyCommand>>>,
//     reporter: Option<ExceptionDispatcher>,
// }
//
// impl Router {
//     pub fn new() -> Self {
//         Self {
//             channels: HashMap::new(),
//         }
//     }
//
//     pub fn install<T>(&mut self, mut driver: T)
//     where
//         T: Driver + 'static,
//     {
//         let (tx, mut rx) = channel(256);
//
//         if self.reporter().is_none() {
//             if let Some(reporter) = driver.reporter() {
//                 self.reporter = Some(reporter.clone());
//             }
//         }
//
//         let type_id = TypeId::of::<T::Command>();
//
//         self.channels.insert(type_id, tx);
//
//         task::spawn(async move {
//             while let Some(op) = rx.recv().await {
//                 match op {
//                     Operation::Create(inst) => driver.create(inst),
//                     Operation::Destroy(inst) => driver.destroy(inst),
//                     Operation::Dispatch(inst, cmd) => {
//                         driver.dispatch(inst, cmd.bx.downcast().unwrap()).await
//                     }
//                 }
//             }
//         });
//     }
// }
//
// impl Driver for Router {
//     type Command = AnyCommand;
//
//     fn create(&mut self, inst: InstanceId) {
//         for (_, sender) in self.channels.iter() {
//             sender.send(Operation::Create(inst)).unwrap();
//         }
//     }
//
//     fn destroy(&mut self, inst: InstanceId) {
//         for (_, sender) in self.channels.iter() {
//             sender.send(Operation::Destroy(inst)).unwrap();
//         }
//     }
//
//     async fn dispatch(&mut self, inst: InstanceId, cmd: Self::Command) {
//         let type_id = cmd.orig_type_id;
//         self.channels
//             .get_mut(&type_id)
//             .ok_or(DriverError::Other("type not found".to_string()))?
//             .send(Operation::Dispatch(inst, cmd))
//             .await
//             .unwrap();
//     }
//
//     fn reporter(&self) -> Option<&ExceptionDispatcher> {
//         self.reporter.as_ref()
//     }
// }
//
// //// ----------------- NameSelector ----------------- ////
// pub struct NamedCommand<N, C>
// where
//     N: Copy + Clone + Eq + Hash + Debug + Send + Sync + 'static,
//     C: Send + Sync + 'static,
// {
//     name: N,
//     cmd: C,
// }
//
// impl<N, C> NamedCommand<N, C>
// where
//     N: Copy + Clone + Eq + Hash + Debug + Send + Sync + 'static,
//     C: Send + Sync + 'static,
// {
//     pub fn new(name: N, cmd: C) -> Self {
//         Self { name, cmd }
//     }
// }
//
// pub struct NameSelector<T, N, C>
// where
//     T: Driver<Command = C>,
//     N: Copy + Clone + Eq + Hash + Debug + Send + Sync + 'static,
//     C: Send + Sync + 'static,
// {
//     phantom: PhantomData<T>,
//     channels: HashMap<N, Sender<Operation<C>>>,
//     handles: HashMap<N, task::JoinHandle<()>>,
//     reporter: Option<ExceptionDispatcher>,
// }
//
// impl<T, N, C> NameSelector<T, N, C>
// where
//     T: Driver<Command = C>,
//     N: Copy + Clone + Eq + Hash + Debug + Send + Sync + 'static,
//     C: Send + Sync + 'static,
// {
//     pub fn new() -> Self {
//         Self {
//             phantom: PhantomData,
//             channels: HashMap::new(),
//             handles: HashMap::new(),
//             reporter: None,
//         }
//     }
//     pub fn with(drivers: Vec<(String, T)>) -> Self {
//         let mut this = Self::new();
//
//         for (name, driver) in drivers {
//             this.add_driver(&name, driver);
//         }
//
//         this
//     }
//
//     pub fn add_driver(&mut self, name: N, mut driver: T) {
//         if self.reporter.is_none() {
//             if let Some(reporter) = driver.reporter() {
//                 self.reporter = Some(reporter.clone());
//             }
//         }
//
//         let (tx, mut rx) = channel(256);
//         let handle = task::spawn(async move {
//             while let Some(op) = rx.recv().await {
//                 match op {
//                     Operation::Create(inst) => driver.create(inst),
//                     Operation::Destroy(inst) => driver.destroy(inst),
//                     Operation::Dispatch(inst, cmd) => driver.dispatch(inst, cmd.cmd).await,
//                 }
//             }
//         });
//
//         self.channels.insert(name.to_string(), tx);
//         self.handles.insert(name.to_string(), handle);
//     }
//
//     pub fn remove_driver(&mut self, name: &str) {
//         self.channels.remove(name);
//         if let Some(handle) = self.handles.remove(name) {
//             handle.abort();
//         }
//     }
// }
//
// impl<T, N, C> Driver for NameSelector<T, N, C>
// where
//     T: Driver<Command = C>,
//     N: Copy + Clone + Eq + Hash + Debug + Send + Sync + 'static,
//     C: Send + Sync + 'static,
// {
//     type Command = NamedCommand<N, C>;
//
//     fn create(&mut self, inst: InstanceId) {
//         for (_, sender) in self.channels.iter() {
//             sender.send(Operation::Create(inst)).unwrap();
//         }
//     }
//
//     fn destroy(&mut self, inst: InstanceId) {
//         for (_, sender) in self.channels.iter() {
//             sender.send(Operation::Destroy(inst)).unwrap();
//         }
//     }
//
//     async fn dispatch(&mut self, inst: InstanceId, cmd: Self::Command) {
//         let NamedCommand { name, cmd } = cmd;
//
//         if let Some(channel) = self.channels.get_mut(&name) {
//             channel.send(Operation::Dispatch(inst, cmd)).await.unwrap();
//         } else {
//             self.reporter()
//                 .map(|r| r.error(inst, "selector name not found".to_string()));
//         }
//     }
//
//     fn reporter(&self) -> Option<&ExceptionDispatcher> {
//         self.reporter.as_ref()
//     }
// }
//
// // TODO
// pub struct LoadBalancer<T, C>
// where
//     T: Driver<Command = C>,
// {
//     drivers: Vec<T>,
//     idx: usize,
// }
