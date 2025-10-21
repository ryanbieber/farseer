// Test std::autodiff reverse-mode AD
#![feature(autodiff)]
#![allow(incomplete_features)]

use std::autodiff::autodiff_reverse;

// Simple quadratic function
#[autodiff_reverse(dquadratic, Dual, Const)]
fn quadratic(x: f64, c: f64) -> f64 {
    x * x + c
}

fn main() {
    // Test the generated gradient function
    let x = 3.0;
    let c = 1.0;
    
    let (value, grad_x) = dquadratic(x, c);
    
    println!("quadratic({}, {}) = {}", x, c, value);
    println!("d/dx quadratic({}, {}) = {}", x, c, grad_x);
    println!("Expected: 2*x = {}", 2.0 * x);
    
    // Test with another value
    let x = 5.0;
    let (value, grad_x) = dquadratic(x, c);
    println!("\nquadratic({}, {}) = {}", x, c, value);
    println!("d/dx quadratic({}, {}) = {}", x, c, grad_x);
    println!("Expected: 2*x = {}", 2.0 * x);
}
