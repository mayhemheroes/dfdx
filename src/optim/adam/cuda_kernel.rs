use crate::{
    optim::optimizer::*,
    shapes::*,
    tensor::{launch_cfg, Cuda},
};

use cudarc::driver::{DeviceRepr, DeviceSlice, LaunchAsync};

#[repr(C)]
struct CudaAdamConfig {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay_type: WeightDecayType,
    weight_decay: f64,
}

unsafe impl DeviceRepr for CudaAdamConfig {}

fn adam_config_to_cuda(config: &super::AdamConfig) -> CudaAdamConfig {
    let (weight_decay_type, weight_decay) = weight_decay_to_cuda(config.weight_decay);

    CudaAdamConfig {
        lr: config.lr,
        beta1: config.betas[0],
        beta2: config.betas[1],
        eps: config.eps,
        weight_decay_type,
        weight_decay,
    }
}

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/adam.ptx"));

trait HasCudaKernel<E> {
    const MOD: &'static str;
    const FWD: &'static str;
}

#[cfg(feature = "f16")]
impl HasCudaKernel<half::f16> for Cuda {
    const MOD: &'static str = "adam_f16";
    const FWD: &'static str = "adam_update_f16";
}

impl HasCudaKernel<f32> for Cuda {
    const MOD: &'static str = "adam_f32";
    const FWD: &'static str = "adam_update_f32";
}

impl HasCudaKernel<f64> for Cuda {
    const MOD: &'static str = "adam_f64";
    const FWD: &'static str = "adam_update_f64";
}

impl<E: Dtype> super::AdamKernel<E> for Cuda
where
    Self: HasCudaKernel<E>,
{
    fn update(
        &self,
        t: i32,
        cfg: &super::AdamConfig,
        param: &mut Self::Vec<E>,
        moment1: &mut Self::Vec<E>,
        moment2: &mut Self::Vec<E>,
        grad: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        if !self.dev.has_func(Self::MOD, Self::FWD) {
            self.dev.load_ptx(PTX_SRC.into(), Self::MOD, &[Self::FWD])?;
        }

        let opt_cfg = adam_config_to_cuda(cfg);
        let numel = param.len();
        let func = self.dev.get_func(Self::MOD, Self::FWD).unwrap();
        let cfg = launch_cfg::<128>(numel as u32);
        let params = (opt_cfg, numel, t, param, moment1, moment2, grad);
        unsafe { func.launch(cfg, params) }?;
        Ok(())
    }
}
