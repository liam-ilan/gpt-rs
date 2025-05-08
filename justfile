default:
    just -l

# Reset libtorch installation.
setup-reset:
    [ ! -e "./libtorch" ] || rm -rf "./libtorch"
    [ ! -e "./.cargo/config.toml" ] || rm "./.cargo/config.toml"

# Install libtorch for macos arm64 and setup config.
setup-macos-arm64: setup-reset
    curl -LO "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.6.0.zip"
    unzip "libtorch-macos-arm64-2.6.0.zip" -d .
    rm "libtorch-macos-arm64-2.6.0.zip"
    
    cp "./.cargo/macos-arm64-config.toml" "./.cargo/config.toml"