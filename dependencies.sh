#!/bin/bash

# Install Homebrew if not already installed (for macOS)
if ! command -v brew &> /dev/null; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install ffmpeg
brew install ffmpeg

# Install PostgreSQL
brew install postgresql

# Start PostgreSQL service
brew services start postgresql

# Create PostgreSQL user and database
psql -U postgres -c "CREATE ROLE hodahashemi WITH LOGIN PASSWORD 'your_password';" || true
psql -U postgres -c "ALTER ROLE hodahashemi CREATEDB;" || true
psql -U postgres -c "CREATE DATABASE my_semantic_db;" || true
psql -U hodahashemi -d my_semantic_db -c "CREATE EXTENSION IF NOT EXISTS vector;" || true

echo "System dependencies installed and PostgreSQL configured."

