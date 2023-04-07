use crate::{shapes::*, tensor::*};

mod cpu_kernel;

pub trait AffineNormalizeKernel<E: Dtype>: DeviceStorage {
    fn forward<S: Shape>(
        &self,
        x: Result<Tensor<S, E, Self>, &Tensor<S, E, Self>>,
        mean: &Tensor<S, E, Self>,
        var: &Tensor<S, E, Self>,
        scale: &Tensor<S, E, Self>,
        bias: &Tensor<S, E, Self>,
        epsilon: E,
    ) -> Result<Tensor<S, E, Self>, Self::Err>;

    #[allow(clippy::too_many_arguments)]
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
    ) -> Result<(), Self::Err>;
}

pub trait TryAffineNormalize<E>: HasErr {
    fn affine_normalize(self, mean: Self, var: Self, scale: Self, bias: Self, epsilon: E) -> Self {
        self.try_affine_normalize(mean, var, scale, bias, epsilon)
            .unwrap()
    }
    fn try_affine_normalize(
        self,
        mean: Self,
        var: Self,
        scale: Self,
        bias: Self,
        epsilon: E,
    ) -> Result<Self, Self::Err>;
}

impl<S: Shape, E: Dtype, D: AffineNormalizeKernel<E>, T: Tape<E, D>> TryAffineNormalize<E>
    for Tensor<S, E, D, T>
{
    fn try_affine_normalize(
        self,
        mean: Self,
        var: Self,
        scale: Self,
        bias: Self,
        epsilon: E,
    ) -> Result<Self, Self::Err> {
        let (inp, tape) = self.split_tape();
        let (mean, tape1) = mean.split_tape();
        let (var, tape2) = var.split_tape();
        let (scale, tape3) = scale.split_tape();
        let (bias, tape4) = bias.split_tape();
        let dev = inp.device.clone();
        let tape = tape.merge(tape1);
        let tape = tape.merge(tape2);
        let tape = tape.merge(tape3);
        let tape = tape.merge(tape4);
        let out = if !T::OWNS_TAPE {
            dev.forward(Ok(inp), &mean, &var, &scale, &bias, epsilon)?
        } else {
            todo!()
        };
        Ok(out.put_tape(tape))
    }
}
