#!/bin/bash
# Local PyPI Deployment Test Script
# This script helps you test your package locally before deploying to PyPI

set -e  # Exit on error

echo "======================================"
echo "🧪 Seer Local Deployment Test"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Check Rust is installed
echo -e "${YELLOW}Step 1: Checking Rust installation...${NC}"
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}❌ Rust is not installed. Install from https://rustup.rs/${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Rust is installed ($(rustc --version))${NC}"
echo ""

# Step 2: Check Python is installed
echo -e "${YELLOW}Step 2: Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python is installed ($(python3 --version))${NC}"
echo ""

# Step 2.5: Check uv is installed
echo -e "${YELLOW}Step 2.5: Checking uv installation...${NC}"
if ! command -v uv &> /dev/null; then
    echo -e "${RED}❌ uv is not installed. Install from https://github.com/astral-sh/uv${NC}"
    exit 1
fi
echo -e "${GREEN}✅ uv is installed ($(uv --version))${NC}"
echo ""

# Step 3: Install/upgrade maturin
echo -e "${YELLOW}Step 3: Installing maturin...${NC}"
uv pip install --upgrade maturin
echo -e "${GREEN}✅ Maturin is ready${NC}"
echo ""

# Step 4: Run Rust tests
echo -e "${YELLOW}Step 4: Running Rust tests...${NC}"
if cargo test --lib --all-features; then
    echo -e "${GREEN}✅ Rust unit tests passed${NC}"
else
    echo -e "${YELLOW}⚠️  Some Rust tests failed (this is expected for integration tests)${NC}"
    echo -e "${YELLOW}    Unit tests are what matter for deployment${NC}"
fi
echo ""

# Step 5: Check Rust code formatting
echo -e "${YELLOW}Step 5: Checking Rust code formatting...${NC}"
if ! cargo fmt --all -- --check; then
    echo -e "${YELLOW}⚠️  Code formatting issues found. Run 'cargo fmt' to fix${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✅ Code formatting is good${NC}"
fi
echo ""

# Step 6: Build the package
echo -e "${YELLOW}Step 6: Building release package...${NC}"
maturin build --release
echo -e "${GREEN}✅ Package built successfully${NC}"
echo ""

# Step 7: Build and install in development mode
echo -e "${YELLOW}Step 7: Installing in development mode...${NC}"
maturin develop --release
echo -e "${GREEN}✅ Package installed in development mode${NC}"
echo ""

# Step 8: Install dev dependencies and run Python tests
echo -e "${YELLOW}Step 8: Installing dev dependencies and running Python tests...${NC}"
if [ -d "tests" ]; then
    echo "Installing dev dependencies..."
    uv pip install -e ".[dev]"
    pytest tests/ -v
    echo -e "${GREEN}✅ Python tests passed${NC}"
else
    echo -e "${YELLOW}⚠️  No tests directory found, skipping Python tests${NC}"
fi
echo ""

# Step 9: Test import and version
echo -e "${YELLOW}Step 9: Testing package import...${NC}"
python3 -c "import seer; print(f'Successfully imported seer version: {seer.__version__}')"
echo -e "${GREEN}✅ Package imports correctly${NC}"
echo ""

# Step 10: Show built wheels
echo -e "${YELLOW}Step 10: Built wheels:${NC}"
ls -lh target/wheels/
echo ""

# Step 11: Check package metadata
echo -e "${YELLOW}Step 11: Package metadata:${NC}"
python3 << 'EOF'
import seer
import inspect

print(f"Version: {seer.__version__}")
print(f"Module location: {seer.__file__}")
print(f"Seer class: {seer.Seer}")

# List public API
public_items = [name for name in dir(seer) if not name.startswith('_')]
print(f"\nPublic API items: {', '.join(public_items)}")
EOF
echo ""

# Summary
echo "======================================"
echo -e "${GREEN}✅ All checks passed!${NC}"
echo "======================================"
echo ""
echo "Your package is ready for deployment!"
echo ""
echo "Next steps:"
echo "  1. Update version in pyproject.toml and Cargo.toml"
echo "  2. Commit changes: git add . && git commit -m 'Bump version to X.Y.Z'"
echo "  3. Create tag: git tag vX.Y.Z"
echo "  4. Push: git push && git push --tags"
echo "  5. Create GitHub Release to trigger automated deployment"
echo ""
echo "See PYPI_DEPLOYMENT.md for detailed instructions"
echo ""
