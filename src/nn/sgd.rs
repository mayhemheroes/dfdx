use super::traits::Optimizer;
use crate::prelude::Tensor;
use crate::prelude::Tensor0D;
use crate::tensor::OnGradientTape;
use std::ops::{Deref, DerefMut};

#[derive(Debug)]
pub struct SgdConfig {
    pub lr: f32,
}

impl Default for SgdConfig {
    fn default() -> Self {
        Self { lr: 1e-2 }
    }
}

#[derive(Default, Debug)]
pub struct Sgd<M> {
    pub cfg: SgdConfig,
    pub module: M,
}

impl<M> Sgd<M>
where
    M: Default,
{
    pub fn with_config(cfg: SgdConfig) -> Self {
        Self {
            cfg,
            module: Default::default(),
        }
    }
}

impl<M> Deref for Sgd<M> {
    type Target = M;
    fn deref(&self) -> &Self::Target {
        &self.module
    }
}

impl<M> DerefMut for Sgd<M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.module
    }
}

impl<M> Optimizer<M> for Sgd<M>
where
    M: OnGradientTape,
{
    fn step(&mut self, loss: &Tensor0D) {
        let mut gradients = loss.backward().unwrap();
        gradients.scale(self.cfg.lr);
        self.module.update_with(&gradients);
    }
}
