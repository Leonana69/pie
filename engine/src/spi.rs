use wasmtime::component::Resource;
use wasmtime::component::{bindgen, ResourceTable};
use wasmtime::Result;
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiView};

use tokio::sync::mpsc::{Receiver, Sender};
use uuid::Uuid;

bindgen!({
    path: "../spi/app/wit",
    world: "app",
    async: true,
    with: {
        "spi:lm/inference/language-model": LanguageModel,
        "spi:lm/kvcache/token": CachedToken,
        "spi:lm/kvcache/token-list": CachedTokenList,
    },
    // Interactions with `ResourceTable` can possibly trap so enable the ability
    // to return traps from generated functions.
    trappable_imports: true,
});

pub struct InstanceState {
    pub instance_id: Uuid,

    pub wasi_ctx: WasiCtx,
    pub resource_table: ResourceTable,

    // For communication between the instance and the host
    pub inst2server: Sender<InstanceMessage>,
    pub server2inst: Receiver<InstanceMessage>,
}

pub struct InstanceMessage {
    pub instance_id: Uuid,
    pub dest_id: u32,
    pub message: String,
}

impl WasiView for InstanceState {
    fn table(&mut self) -> &mut ResourceTable {
        &mut self.resource_table
    }
    fn ctx(&mut self) -> &mut WasiCtx {
        &mut self.wasi_ctx
    }
}

impl InstanceState {
    pub fn new(
        instance_id: Uuid,
        inst2server: Sender<InstanceMessage>,
        server2inst: Receiver<InstanceMessage>,
    ) -> Self {
        let mut builder = WasiCtx::builder();
        builder.inherit_stderr().inherit_network().inherit_stdout();

        InstanceState {
            instance_id,
            wasi_ctx: builder.build(),
            resource_table: ResourceTable::new(),
            inst2server,
            server2inst,
        }
    }
}

pub struct LanguageModel {
    model_id: String,
}

#[derive(Clone, Copy)]
pub struct CachedToken {
    token: u32,
    id: u32,
}

pub struct CachedTokenList {
    tokens: Vec<CachedToken>,
}

impl spi::lm::inference::Host for InstanceState {}
impl spi::lm::inference::HostLanguageModel for InstanceState {
    async fn new(&mut self, model_id: String) -> Result<Resource<LanguageModel>, wasmtime::Error> {
        let handle = LanguageModel { model_id };
        Ok(self.resource_table.push(handle)?)
    }

    async fn tokenize(
        &mut self,
        resource: Resource<LanguageModel>,
        text: String,
    ) -> Result<Vec<u32>, wasmtime::Error> {
        Ok(vec![0])
    }

    async fn detokenize(
        &mut self,
        resource: Resource<LanguageModel>,
        tokens: Vec<u32>,
    ) -> Result<String, wasmtime::Error> {
        Ok("Hello".to_string())
    }

    async fn predict(
        &mut self,
        resource: Resource<LanguageModel>,
        tokens: Vec<u32>,
    ) -> Result<u32, wasmtime::Error> {
        Ok(7)
    }

    async fn drop(&mut self, resource: Resource<LanguageModel>) -> Result<()> {
        let _ = self.resource_table.delete(resource)?;

        Ok(())
    }
}
//
impl spi::app::system::Host for InstanceState {
    async fn get_version(&mut self) -> Result<String, wasmtime::Error> {
        Ok("0.1.0".to_string())
    }

    async fn send(&mut self, dest_id: u32, message: String) -> Result<()> {
        self.inst2server
            .send(InstanceMessage {
                instance_id: self.instance_id,
                dest_id,
                message,
            })
            .await?;

        Ok(())
    }

    async fn receive(&mut self) -> Result<String, wasmtime::Error> {
        if let Some(message) = self.server2inst.recv().await {
            return Ok(message.message);
        }

        Ok("".to_string())
    }
}

impl spi::lm::kvcache::Host for InstanceState {}

impl spi::lm::kvcache::HostToken for InstanceState {
    async fn position(&mut self, resource: Resource<CachedToken>) -> Result<u32, wasmtime::Error> {
        Ok(0)
    }

    async fn token_id(&mut self, resource: Resource<CachedToken>) -> Result<u32, wasmtime::Error> {
        Ok(1)
    }

    async fn drop(&mut self, resource: Resource<CachedToken>) -> Result<()> {
        let _ = self.resource_table.delete(resource)?;
        Ok(())
    }
}

impl spi::lm::kvcache::HostTokenList for InstanceState {
    //
    //
    /*
        constructor(tokens: list<token>);

    // mutating methods
    push: func(token: token);
    pop: func() -> token;
    extend: func(tokens: token-list);
    splice: func(start: u32, delete-count: u32, tokens: token-list);

    // non-mutating methods
    length: func() -> u32;
    slice: func(start: u32, end: u32) -> token-list;
    concat: func(cache: token-list) -> token-list;
    index: func(position: u32) -> token;
     */

    //
    // 1) Constructor
    //
    //    WIT signature (approx):
    //    constructor(tokens: list<token>) -> token-list
    //
    //    * `tokens` here is a Vec<Resource<CachedToken>> from the generated code.
    //
    async fn new(
        &mut self,
        tokens: Vec<Resource<CachedToken>>,
    ) -> Result<Resource<CachedTokenList>> {
        // Collect actual CachedToken data from each resource
        let mut list_data = Vec::with_capacity(tokens.len());
        for token_resource in tokens {
            // Get a reference to the cached token from the table:
            let token_ref = self.resource_table.get(&token_resource)?;
            // Clone it if you plan to store a copy
            list_data.push(token_ref.clone());
        }

        // Create a new CachedTokenList resource
        let token_list = CachedTokenList { tokens: list_data };
        let resource_handle = self.resource_table.push(token_list)?;
        Ok(resource_handle)
    }

    //
    // 2) push: func(token: token);
    //
    //    * Mutates the list by pushing a new token onto it.
    //    * The `token` is a Resource<CachedToken>.
    //
    async fn push(
        &mut self,
        list_resource: Resource<CachedTokenList>,
        token_resource: Resource<CachedToken>,
    ) -> Result<()> {
        let token_ref = self.resource_table.get(&token_resource)?.clone();
        let token_list = self.resource_table.get_mut(&list_resource)?;
        token_list.tokens.push(token_ref.clone());
        Ok(())
    }

    //
    // 3) pop: func() -> token;
    //
    //    * Mutates the list by popping the last token
    //      and returns it as a fresh Resource<CachedToken>.
    //
    async fn pop(
        &mut self,
        list_resource: Resource<CachedTokenList>,
    ) -> Result<Resource<CachedToken>> {
        let token_list = self.resource_table.get_mut(&list_resource)?;

        let popped = token_list
            .tokens
            .pop()
            .ok_or_else(|| anyhow::anyhow!("Cannot pop from an empty list"))?;

        // Push the popped token into the resource table so the caller can use it
        let popped_resource = self.resource_table.push(popped)?;
        Ok(popped_resource)
    }

    //
    // 4) extend: func(tokens: token-list);
    //
    //    * Mutates the `list_resource` by extending with all the tokens
    //      from `other_list_resource`.
    //    * We do NOT remove them from the `other_list`; we just copy them.
    //
    async fn extend(
        &mut self,
        list_resource: Resource<CachedTokenList>,
        other_list_resource: Resource<CachedTokenList>,
    ) -> Result<()> {
        let token_list = self.resource_table.get_mut(&list_resource)?;
        let other_list = self.resource_table.get(&other_list_resource)?;

        // Copy the tokens from other_list
        token_list.tokens.extend(other_list.tokens.clone());
        Ok(())
    }

    //
    // 5) splice: func(start: u32, delete_count: u32, tokens: token-list);
    //
    //    * Removes `delete_count` items from `list_resource` starting at `start`
    //      and inserts tokens from `other_list_resource` in their place.
    //
    async fn splice(
        &mut self,
        list_resource: Resource<CachedTokenList>,
        start: u32,
        delete_count: u32,
        other_list_resource: Resource<CachedTokenList>,
    ) -> Result<()> {
        let token_list = self.resource_table.get_mut(&list_resource)?;
        let other_list = self.resource_table.get(&other_list_resource)?;

        let start_usize = start as usize;
        let delete_count_usize = delete_count as usize;

        // Safety check or clamp if needed:
        let len = token_list.tokens.len();
        if start_usize > len {
            return Err(anyhow::anyhow!("Start index out of bounds"));
        }
        let end_delete = (start_usize + delete_count_usize).min(len);

        // Splice out delete_count items and replace with `other_list.tokens`
        token_list
            .tokens
            .splice(start_usize..end_delete, other_list.tokens.clone());
        Ok(())
    }

    //
    // 6) length: func() -> u32;
    //
    async fn length(&mut self, list_resource: Resource<CachedTokenList>) -> Result<u32> {
        let token_list = self.resource_table.get(&list_resource)?;
        Ok(token_list.tokens.len() as u32)
    }

    //
    // 7) slice: func(start: u32, end: u32) -> token-list;
    //
    //    * Returns a new token-list resource with tokens from [start..end).
    //
    async fn slice(
        &mut self,
        list_resource: Resource<CachedTokenList>,
        start: u32,
        end: u32,
    ) -> Result<Resource<CachedTokenList>> {
        let token_list = self.resource_table.get(&list_resource)?;
        let len = token_list.tokens.len();

        let start_usize = start as usize;
        let end_usize = end as usize;
        if start_usize > len || end_usize > len || start_usize > end_usize {
            return Err(anyhow::anyhow!("Invalid slice range"));
        }

        let new_tokens = token_list.tokens[start_usize..end_usize].to_vec();
        let new_list = CachedTokenList { tokens: new_tokens };
        let handle = self.resource_table.push(new_list)?;
        Ok(handle)
    }

    //
    // 8) concat: func(cache: token-list) -> token-list;
    //
    //    * Returns a new token-list that is `list_resource` + `other_list_resource`.
    //
    async fn concat(
        &mut self,
        list_resource: Resource<CachedTokenList>,
        other_list_resource: Resource<CachedTokenList>,
    ) -> Result<Resource<CachedTokenList>> {
        let token_list = self.resource_table.get(&list_resource)?;
        let other_list = self.resource_table.get(&other_list_resource)?;

        let mut new_vec = token_list.tokens.clone();
        new_vec.extend(other_list.tokens.clone());

        let new_list = CachedTokenList { tokens: new_vec };
        let handle = self.resource_table.push(new_list)?;
        Ok(handle)
    }

    //
    // 9) index: func(position: u32) -> token;
    //
    //    * Returns the token at `position` in `list_resource`.
    //      We wrap it in a new Resource<CachedToken> so that the caller
    //      can manipulate it.
    //
    async fn index(
        &mut self,
        list_resource: Resource<CachedTokenList>,
        position: u32,
    ) -> Result<Resource<CachedToken>> {
        let token_list = self.resource_table.get(&list_resource)?;
        let idx = position as usize;
        if idx >= token_list.tokens.len() {
            return Err(anyhow::anyhow!("Index out of range"));
        }

        let token = token_list.tokens[idx].clone();
        // Put this token into the resource table and return that resource
        let token_resource = self.resource_table.push(token)?;
        Ok(token_resource)
    }

    async fn drop(&mut self, resource: Resource<CachedTokenList>) -> Result<()> {
        let _ = self.resource_table.delete(resource)?;
        Ok(())
    }
}
