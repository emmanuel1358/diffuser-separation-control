# OpenROAD/EDA GPU overlay.
#
# The default gpu flavor is a clean ML/H100 image. This flavor opts EDA tasks
# into OpenROAD's Ubuntu-built binary tree and runtime ABI libraries.
ARG BASE_IMAGE=lbx-tasks-base-gpu
ARG BASE_TAG=runtime-ml-core-py313-local
ARG OPENROAD_PLATFORM=linux/amd64

FROM --platform=${OPENROAD_PLATFORM} openroad/orfs:latest AS openroad
FROM ${BASE_IMAGE}:${BASE_TAG}

ENV PATH="/opt/openroad/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/openroad/lib:/opt/or-tools/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64"

# OpenROAD binary tree + its bundled solver libs (or-tools/scip/highs/abseil).
COPY --from=openroad /OpenROAD-flow-scripts/tools/install/OpenROAD /opt/openroad
COPY --from=openroad /opt/or-tools /opt/or-tools

# Runtime system libs OpenROAD's Ubuntu-22.04 binary links against, plus a
# stable CLI symlink.
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libbrotli1 libbsd0 libbz2-1.0 libdouble-conversion3 libexpat1 \
      libfreetype6 libgl1 libglib2.0-0 libglvnd0 libglx0 libgomp1 \
      libgraphite2-3 libharfbuzz0b libicu70 libmd0 libmd4c0 \
      libpcre2-16-0 libpcre3 libpng16-16 libpython3.10 \
      libqt5charts5 libqt5core5a libqt5gui5 libqt5widgets5 \
      libtcl8.6 libx11-6 libxau6 libxcb1 libxdmcp6 libyaml-cpp0.7 libzstd1 \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /opt/openroad/bin/openroad /usr/local/bin/openroad \
 && openroad -version

CMD ["/opt/lbx-runtime/.venv/bin/rubric", "mcp"]
