pub use crate::api::exports::inferlet::core::run::Guest;
pub use crate::api::inferlet::adapter;
pub use crate::api::inferlet::core::common::{
    Blob, BlobResult, DebugQueryResult, Model, Queue, SynchronizationResult, Priority,
    allocate_resources,
    deallocate_resources,
    export_resources,
    import_resources,
    get_all_exported_resources,
    release_exported_resources
};
pub use crate::api::inferlet::core::forward;
pub use crate::api::inferlet::core::kvs;
pub use crate::api::inferlet::core::message;
pub use crate::api::inferlet::core::runtime;
pub use crate::api::inferlet::core::tokenize;
pub use crate::api::inferlet::image;
pub use crate::api::inferlet::zo;

wit_bindgen::generate!({
    path: "wit",
    world: "exec",
    pub_export_macro: true,
    with: {
         "wasi:io/poll@0.2.4": wasi::io::poll,
    },
    generate_all,
});
