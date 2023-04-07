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
        todo!()
    }
}
