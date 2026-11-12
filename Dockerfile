################## BASE IMAGE ######################
FROM python:3.10-slim

################## METADATA ######################
LABEL base_image="python:3.10-slim"
LABEL version="1"
LABEL software="pyspectrafuse"
LABEL software.version="0.0.6"
LABEL about.summary="pyspectrafuse - CLI for spectral clustering and consensus library generation"
LABEL about.home="https://github.com/bigbio/pyspectrafuse"
LABEL about.documentation="https://github.com/bigbio/pyspectrafuse"
LABEL about.license_file="https://github.com/bigbio/pyspectrafuse/blob/master/LICENSE"
LABEL about.license="SPDX:Apache-2.0"
LABEL about.tags="Proteomics,Multiomics,QuantMS"

################## MAINTAINER ######################
MAINTAINER Yasset Perez-Riverol <ypriverol@gmail.com>

################## INSTALLATION ######################

ENV DEBIAN_FRONTEND=noninteractive

# Disable numba caching to avoid issues in containerized environments
ENV NUMBA_DISABLE_CACHING=1

## Update and install packages
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    procps \
    && rm -rf /var/lib/apt/lists/*

## Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

## Set working directory
WORKDIR /data/

## Copy project files
COPY pyproject.toml .
COPY README.md .
COPY pyspectrafuse/ ./pyspectrafuse/

## Install the package with uv
RUN uv pip install --system --no-cache .

## No ENTRYPOINT — Nextflow runs commands via /bin/bash
## Use `pyspectrafuse` as the CLI command in scripts
CMD ["pyspectrafuse", "--help"]
