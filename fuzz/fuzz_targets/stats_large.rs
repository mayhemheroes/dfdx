#![no_main]
use dfdx::{tensor::{Cpu, TensorFrom}, prelude::{mae_loss, mse_loss, rmse_loss, smooth_l1_loss}};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|input: (Vec<f64>, Vec<f64>, f64)| {
    let (lhs_vals, rhs_vals, delta) = input;
    for i in lhs_vals.iter() {
        if i.is_nan() {
            return
        }
    };
    for i in rhs_vals.iter() {
        if i.is_nan() {
            return
        }
    };
    if delta.is_nan() {
        return;
    }


    let lhs_min = lhs_vals.len().min(256);
    let mut lhs_vec = [0.0; 256];
    lhs_vec[0..lhs_min].copy_from_slice(&lhs_vals[0..lhs_min]);

    let rhs_min = rhs_vals.len().min(256);
    let mut rhs_vec = [0.0; 256];
    rhs_vec[0..rhs_min].copy_from_slice(&rhs_vals[0..rhs_min]);

    let dev: Cpu = Default::default();
    let lhs_tensor = dev.tensor(lhs_vec);
    let rhs_tensor = dev.tensor(rhs_vec);

    lhs_tensor.clone().huber_error(rhs_tensor.clone(), delta);
    mae_loss(lhs_tensor.clone(), rhs_tensor.clone());
    mse_loss(lhs_tensor.clone(), rhs_tensor.clone());
    rmse_loss(lhs_tensor.clone(), rhs_tensor.clone());
    smooth_l1_loss(lhs_tensor.clone(), rhs_tensor.clone(), delta);
});