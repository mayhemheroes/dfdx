use crate::{shapes::*, tensor::*};

mod cpu_kernel;

pub trait AffineNormalizeKernel<E: Dtype>: DeviceStorage {
    fn forward<S: Shape>(
        &self,
        x: Result<Tensor<S, E, Self>, &Tensor<S, E, Self>>,
        mean: Option<&Tensor<S, E, Self>>,
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
        mean: Option<&Tensor<S, E, Self>>,
        grad_mean: Option<&mut Self::Vec<E>>,
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
    fn affine_normalize(
        self,
        mean: Option<Self>,
        var: Self,
        scale: Self,
        bias: Self,
        epsilon: E,
    ) -> Self {
        self.try_affine_normalize(mean, var, scale, bias, epsilon)
            .unwrap()
    }
    fn try_affine_normalize(
        self,
        mean: Option<Self>,
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
        mean: Option<Self>,
        var: Self,
        scale: Self,
        bias: Self,
        epsilon: E,
    ) -> Result<Self, Self::Err> {
        let (inp, mut tape) = self.split_tape();
        let dev = inp.device.clone();
        let (mean, tape1) = mean.map(|m| m.split_tape()).unzip();
        if let Some(tape1) = tape1 {
            tape = tape.merge(tape1);
        }
        let (var, tape2) = var.split_tape();
        tape = tape.merge(tape2);
        let (scale, tape3) = scale.split_tape();
        tape = tape.merge(tape3);
        let (bias, tape4) = bias.split_tape();
        tape = tape.merge(tape4);

        let out = if !T::OWNS_TAPE {
            dev.forward(Ok(inp), mean.as_ref(), &var, &scale, &bias, epsilon)?
        } else {
            let out = dev.forward(Err(&inp), mean.as_ref(), &var, &scale, &bias, epsilon)?;
            let inp_gh = inp.ghost();
            let mean_gh = mean.as_ref().map(|m| m.ghost());
            let var_gh = var.ghost();
            let scale_gh = scale.ghost();
            let bias_gh = bias.ghost();
            let out_gh = out.ghost();
            tape.add_backward_op(move |grads| {
                grads.try_alloc_for(&inp_gh)?;
                if let Some(mean_ghost) = &mean_gh {
                    grads.try_alloc_for(mean_ghost)?;
                }
                grads.try_alloc_for(&var_gh)?;
                grads.try_alloc_for(&scale_gh)?;
                grads.try_alloc_for(&bias_gh)?;
                grads.try_alloc_for(&out_gh)?;
                let (inps, grad_out) = match &mean_gh {
                    Some(mean_ghost) => {
                        let (inps, grad_out) = grads.many_and_ref(
                            &[&inp_gh, mean_ghost, &var_gh, &scale_gh, &bias_gh],
                            &out_gh,
                        );
                        let inps: [&mut D::Vec<E>; 5] = inps.try_into().unwrap();
                        let [grad_x, grad_mean, grad_var, grad_scale, grad_bias] = inps;
                        (
                            (grad_x, Some(grad_mean), grad_var, grad_scale, grad_bias),
                            grad_out,
                        )
                    }
                    None => {
                        let (inps, grad_out) =
                            grads.many_and_ref(&[&inp_gh, &var_gh, &scale_gh, &bias_gh], &out_gh);
                        let inps: [&mut D::Vec<E>; 4] = inps.try_into().unwrap();
                        let [grad_x, grad_var, grad_scale, grad_bias] = inps;
                        ((grad_x, None, grad_var, grad_scale, grad_bias), grad_out)
                    }
                };
                let (grad_x, grad_mean, grad_var, grad_scale, grad_bias) = inps;
                inp_gh.dev.backward(
                    &inp,
                    grad_x,
                    mean.as_ref(),
                    grad_mean,
                    &var,
                    grad_var,
                    &scale,
                    grad_scale,
                    &bias,
                    grad_bias,
                    epsilon,
                    grad_out,
                )
            });
            out
        };
        Ok(out.put_tape(tape))
    }
}
