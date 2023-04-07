use crate::{shapes::*, tensor::cpu::NdIndex, tensor::*, tensor_ops::ReshapeTo};

impl<E: Dtype + num_traits::Float> super::AffineNormalizeKernel<E> for Cpu {
    fn forward<S: Shape>(
        &self,
        x: Result<Tensor<S, E, Self>, &Tensor<S, E, Self>>,
        opt_mean: Option<&Tensor<S, E, Self>>,
        var: &Tensor<S, E, Self>,
        scale: &Tensor<S, E, Self>,
        bias: &Tensor<S, E, Self>,
        epsilon: E,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        let shape = var.shape;
        let mut out = match x {
            Ok(mut out) if out.strides == out.shape.strides() => {
                out.id = unique_id();
                out
            }
            Ok(inp) => inp.try_reshape_like(&shape).unwrap()?,
            Err(inp) if inp.strides == inp.shape.strides() => {
                let mut out = inp.clone().try_reshape_like(&shape).unwrap()?;
                out.id = unique_id();
                out
            }
            Err(inp) => inp.clone().try_reshape_like(&shape).unwrap()?,
        };

        let mut var_idx = NdIndex::new(var.shape, var.strides);
        let mut scale_idx = NdIndex::new(scale.shape, scale.strides);
        let mut bias_idx = NdIndex::new(bias.shape, bias.strides);
        let mut opt_mean_idx = opt_mean.map(|mean| NdIndex::new(mean.shape, mean.strides));
        for y_i in out.buf_iter_mut() {
            let mean_i = opt_mean_idx
                .as_mut()
                .map(|idx| idx.next().unwrap())
                .map(|i_mean| opt_mean.unwrap().data[i_mean])
                .unwrap_or_default();
            let var_i = var.data[var_idx.next().unwrap()];
            let scale_i = scale.data[scale_idx.next().unwrap()];
            let bias_i = bias.data[bias_idx.next().unwrap()];
            let std_i = (var_i + epsilon).sqrt();
            *y_i = (*y_i - mean_i) * scale_i / std_i + bias_i;
        }
        Ok(out)
    }

    fn backward<S: Shape>(
        &self,
        x: &Tensor<S, E, Self>,
        grad_x: &mut Self::Vec<E>,
        opt_mean: Option<&Tensor<S, E, Self>>,
        mut opt_grad_mean: Option<&mut Self::Vec<E>>,
        var: &Tensor<S, E, Self>,
        grad_var: &mut Self::Vec<E>,
        scale: &Tensor<S, E, Self>,
        grad_scale: &mut Self::Vec<E>,
        bias: &Tensor<S, E, Self>,
        grad_bias: &mut Self::Vec<E>,
        epsilon: E,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let mut opt_mean_idx = opt_mean.map(|mean| NdIndex::new(mean.shape, mean.strides));
        let mut var_idx = NdIndex::new(var.shape, var.strides);
        let mut scale_idx = NdIndex::new(scale.shape, scale.strides);
        let mut bias_idx = NdIndex::new(bias.shape, bias.strides);
        let mut x_idx = NdIndex::new(x.shape, x.strides);
        for &gy in grad_out.iter() {
            let i_x = x_idx.next().unwrap();
            let opt_i_mean = opt_mean_idx.as_mut().map(|idx| idx.next().unwrap());
            let i_var = var_idx.next().unwrap();
            let i_scale = scale_idx.next().unwrap();
            let i_bias = bias_idx.next().unwrap();

            let x_i = x.data[i_x];
            let var_i = var.data[i_var];
            let scale_i = scale.data[i_scale];
            let std_i = (var_i + epsilon).sqrt();
            let v = (var_i + epsilon).powf(E::from_f32(1.5).unwrap());

            let mean_i = match opt_i_mean {
                Some(i_mean) => {
                    opt_grad_mean.as_mut().unwrap()[i_mean] -= gy * scale_i / std_i;
                    opt_mean.as_ref().unwrap().data[i_mean]
                }
                None => Default::default(),
            };
            let centered_i = x_i - mean_i;

            grad_x[i_x] += gy * scale_i / std_i;
            grad_var[i_var] -= gy * centered_i * scale_i / (v + v);
            grad_scale[i_scale] += gy * centered_i / std_i;
            grad_bias[i_bias] += gy;
        }
        Ok(())
    }
}
