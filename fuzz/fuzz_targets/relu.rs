#![no_main]
use dfdx::{tensor::{Cpu, TensorFrom}, tensor_ops::{TryPReLU, leakyrelu}};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|input: (Vec<f64>, f64)| {
    let (tensor_vals, rhs) = input;
    for i in tensor_vals.iter() {
        if i.is_nan() {
            return
        }
    };

    let min = tensor_vals.len().min(32);
    let mut tensor_vec = [0.0; 32];
    tensor_vec[0..min].copy_from_slice(&tensor_vals[0..min]);

    let dev: Cpu = Default::default();
    let tensor = dev.tensor(tensor_vec);

    tensor.clone().relu();
    tensor.clone().prelu(rhs);
    leakyrelu(tensor, rhs);
});