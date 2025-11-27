#!/bin/bash

echo "🧹 Windows Disk Cleanup"

echo "1. Cleaning Windows temp files..."
rm -rf /c/Windows/Temp/* 2>/dev/null || true
rm -rf /c/Users/$USERNAME/AppData/Local/Temp/* 2>/dev/null || true

echo "2. Cleaning browser caches..."
rm -rf "/c/Users/$USERNAME/AppData/Local/Google/Chrome/User Data/Default/Cache/*" 2>/dev/null || true
rm -rf "/c/Users/$USERNAME/AppData/Local/Microsoft/Edge/User Data/Default/Cache/*" 2>/dev/null || true

echo "3. Cleaning Python and project caches..."
find /c/Users/$USERNAME -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find /c/Users/$USERNAME -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

echo "4. Checking current disk space..."
df -h

echo "✅ Cleanup complete!"
