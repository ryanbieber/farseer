// Test BridgeStan integration
use bridgestan;

fn main() {
    println!("BridgeStan version: Check if we can import it");
    
    // First, let's see what's available in the bridgestan crate
    println!("BridgeStan crate imported successfully");
    
    // The basic workflow for BridgeStan:
    // 1. Compile .stan file to .so library
    // 2. Load the compiled model
    // 3. Create data (JSON format)
    // 4. Run optimization or sampling
    // 5. Extract parameters
    
    println!("\nNext steps:");
    println!("1. Compile prophet.stan to a shared library");
    println!("2. Load model using BridgeStan::Model");
    println!("3. Prepare data in JSON format matching Stan data block");
    println!("4. Run optimization to get MAP estimates");
}
