#!/usr/bin/env bash
# exit on error
set -o errexit

# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Build Frontend
echo "📦 Building Frontend..."
cd web
npm install
npm run build
cd ..

echo "✅ Build Complete!"
