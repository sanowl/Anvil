use std::sync::Arc;
use tokio::runtime::Runtime;
use async_trait::async_trait;

#[async_trait]
pub trait AsyncExecutor: Send + Sync {
    fn spawn<F>(&self, future: F)
    where
        F: std::future::Future + Send + 'static,
        F::Output: Send + 'static;
}

pub struct TokioExecutor {
    runtime: Arc<Runtime>,
}

impl TokioExecutor {
    pub fn new() -> Self {
        Self {
            runtime: Arc::new(Runtime::new().unwrap()),
        }
    }
}

impl AsyncExecutor for TokioExecutor {
    fn spawn<F>(&self, future: F)
    where
        F: std::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.runtime.spawn(future);
    }
} 