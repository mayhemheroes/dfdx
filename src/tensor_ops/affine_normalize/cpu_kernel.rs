use crate::{shapes::*, tensor::cpu::NdIndex, tensor::*};

impl<E: Dtype + num_traits::Float> super::AffineNormalizeKernel<E> for Cpu {
    fn forward<S: Shape>(
        &self,
        x: Result<Tensor<S, E, Self>, &Tensor<S, E, Self>>,
        mean: &Tensor<S, E, Self>,
        var: &Tensor<S, E, Self>,
        scale: &Tensor<S, E, Self>,
        bias: &Tensor<S, E, Self>,
        epsilon: E,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        let mut mean_idx = NdIndex::new(mean.shape, mean.strides);
        let mut var_idx = NdIndex::new(var.shape, var.strides);
        let mut scale_idx = NdIndex::new(scale.shape, scale.strides);
        let mut bias_idx = NdIndex::new(bias.shape, bias.strides);
        match x {
            Ok(mut out) if out.strides == out.shape.strides() => {
                out.id = unique_id();
                for y_i in out.buf_iter_mut() {
                    let mean_i = mean.data[mean_idx.next().unwrap()];
                    let var_i = var.data[var_idx.next().unwrap()];
                    let scale_i = scale.data[scale_idx.next().unwrap()];
                    let bias_i = bias.data[bias_idx.next().unwrap()];
                    let std_i = (var_i + epsilon).sqrt();
                    *y_i = (*y_i - mean_i) * scale_i / std_i + bias_i;
                }
                Ok(out)
            }
            Ok(inp) => {
                let mut out = self.try_zeros_like(&inp.shape)?;
                let mut x_idx = NdIndex::new(inp.shape, inp.strides);
                for y_i in out.buf_iter_mut() {
                    let x_i = inp.data[x_idx.next().unwrap()];
                    let mean_i = mean.data[mean_idx.next().unwrap()];
                    let var_i = var.data[var_idx.next().unwrap()];
                    let scale_i = scale.data[scale_idx.next().unwrap()];
                    let bias_i = bias.data[bias_idx.next().unwrap()];
                    let std_i = (var_i + epsilon).sqrt();
                    *y_i = (x_i - mean_i) * scale_i / std_i + bias_i;
                }
                Ok(out)
            }
            Err(inp) => {
                let mut out = self.try_zeros_like(&inp.shape)?;
                let mut x_idx = NdIndex::new(inp.shape, inp.strides);
                for y_i in out.buf_iter_mut() {
                    let x_i = inp.data[x_idx.next().unwrap()];
                    let mean_i = mean.data[mean_idx.next().unwrap()];
                    let var_i = var.data[var_idx.next().unwrap()];
                    let scale_i = scale.data[scale_idx.next().unwrap()];
                    let bias_i = bias.data[bias_idx.next().unwrap()];
                    let std_i = (var_i + epsilon).sqrt();
                    *y_i = (x_i - mean_i) * scale_i / std_i + bias_i;
                }
                Ok(out)
            }
        }
    }

    fn backward<S: Shape>(
        &self,
        x: &Tensor<S, E, Self>,
        grad_x: &mut Self::Vec<E>,
        mean: &Tensor<S, E, Self>,
        grad_mean: &mut Self::Vec<E>,
        var: &Tensor<S, E, Self>,
        grad_var: &mut Self::Vec<E>,
        scale: &Tensor<S, E, Self>,
        grad_scale: &mut Self::Vec<E>,
        bias: &Tensor<S, E, Self>,
        grad_bias: &mut Self::Vec<E>,
        epsilon: E,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let mut mean_idx = NdIndex::new(mean.shape, mean.strides);
        let mut var_idx = NdIndex::new(var.shape, var.strides);
        let mut scale_idx = NdIndex::new(scale.shape, scale.strides);
        let mut bias_idx = NdIndex::new(bias.shape, bias.strides);
        let mut x_idx = NdIndex::new(x.shape, x.strides);
        for &gy in grad_out.iter() {
            let i_x = x_idx.next().unwrap();
            let i_mean = mean_idx.next().unwrap();
            let i_var = var_idx.next().unwrap();
            let i_scale = scale_idx.next().unwrap();
            let i_bias = bias_idx.next().unwrap();

            let x_i = x.data[i_x];
            let mean_i = mean.data[i_mean];
            let var_i = var.data[i_var];
            let scale_i = scale.data[i_scale];

            let centered_i = x_i - mean_i;
            let std_i = (var_i + epsilon).sqrt();
            let v = (var_i + epsilon).powf(E::from_f32(1.5).unwrap());

            grad_x[i_x] += gy * scale_i / std_i;
            grad_mean[i_mean] -= gy * scale_i / std_i;
            grad_var[i_var] -= gy * centered_i * scale_i / (v + v);
            grad_scale[i_scale] += gy * centered_i / std_i;
            grad_bias[i_bias] += gy;
        }
        Ok(())
    }
}
