#!/bin/sh
# Point apt at one dated archive snapshot so every image build resolves identical packages.
# Usage: pin-apt.sh YYYYMMDDTHHMMSSZ
set -eu
snapshot="$1"
. /etc/os-release
rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*
# Snapshot Release files are signed but intentionally stale.
echo 'Acquire::Check-Valid-Until "false";' > /etc/apt/apt.conf.d/99snapshot
case "$ID" in
ubuntu)
    archive="https://snapshot.ubuntu.com/ubuntu/$snapshot"
    for suite in "$VERSION_CODENAME" "$VERSION_CODENAME-updates" "$VERSION_CODENAME-security"; do
        echo "deb $archive $suite main universe"
    done > /etc/apt/sources.list
    # The snapshot service is HTTPS-only and the base image has no CA bundle yet. apt still
    # verifies the signed Release files and package hashes, so fetch ca-certificates first.
    apt-get -o Acquire::https::Verify-Peer=false update
    apt-get -o Acquire::https::Verify-Peer=false install -y --no-install-recommends ca-certificates
    ;;
debian)
    {
        echo "deb http://snapshot.debian.org/archive/debian/$snapshot $VERSION_CODENAME $VERSION_CODENAME-updates main"
        echo "deb http://snapshot.debian.org/archive/debian-security/$snapshot $VERSION_CODENAME-security main"
    } > /etc/apt/sources.list
    apt-get update
    apt-get install -y --no-install-recommends ca-certificates
    ;;
*)
    echo "pin-apt.sh: unsupported base distribution $ID" >&2
    exit 1
    ;;
esac
