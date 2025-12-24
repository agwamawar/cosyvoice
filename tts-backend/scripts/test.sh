#!/bin/bash
# =============================================================================
# Test Runner Script
# =============================================================================
# Usage:
#   ./scripts/test.sh           # Run all tests
#   ./scripts/test.sh unit      # Run unit tests only
#   ./scripts/test.sh integration # Run integration tests only
#   ./scripts/test.sh coverage  # Run with coverage report
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  TTS Backend Test Runner${NC}"
echo -e "${GREEN}========================================${NC}"

case "$1" in
  unit)
    echo -e "${YELLOW}Running unit tests...${NC}"
    pytest tests/unit -v
    ;;
  integration)
    echo -e "${YELLOW}Running integration tests...${NC}"
    pytest tests/integration -v -m integration
    ;;
  coverage)
    echo -e "${YELLOW}Running tests with coverage...${NC}"
    pytest --cov=src --cov-report=html --cov-report=term-missing
    echo -e "${GREEN}Coverage report generated in htmlcov/${NC}"
    ;;
  fast)
    echo -e "${YELLOW}Running fast tests (excluding slow)...${NC}"
    pytest tests/unit tests/integration -v -m "not slow"
    ;;
  *)
    echo -e "${YELLOW}Running all tests...${NC}"
    pytest tests/unit tests/integration -v
    ;;
esac

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Tests completed!${NC}"
echo -e "${GREEN}========================================${NC}"
