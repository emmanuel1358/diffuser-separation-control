# OpenROAD/EDA GPU overlay.
#
# The default gpu flavor is a clean ML/H100 image. This flavor opts EDA tasks
# into OpenROAD's Ubuntu-built binary tree and runtime ABI libraries, so normal
# ML tasks do not inherit the EDA ABI stack.
#
# The source image must match base/gpu's Ubuntu release, because nothing is
# rebuilt here -- a prebuilt binary tree is copied in and then linked against
# the OS packages installed below. base/gpu is Ubuntu 24.04 (noble) as of the
# CUDA 13 migration, and this overlay used to copy from openroad/orfs:latest,
# which is still a 22.04 (jammy) build (its image config carries
# org.opencontainers.image.version=22.04). That combination is unfixable by
# renaming packages: the jammy binaries need libicu.so.70,
# libpython3.10.so.1.0 and libyaml-cpp.so.0.7 specifically, and noble ships
# only libicu.so.74, libpython3.12.so.1.0 and libyaml-cpp.so.0.8. The apt list
# could not even resolve (no libicu70/libpython3.10/libyaml-cpp0.7 in noble,
# and Qt5 went through the t64 rename).
#
# openroad/flow-ubuntu24.04-builder is the noble build of the same
# OpenROAD-flow-scripts image, published by the same CI job on the same cadence
# as openroad/orfs (175 tags; both :latest pushed within a minute of each
# other). Its tree layout is identical -- same /OpenROAD-flow-scripts
# WORKDIR, same tools/install/OpenROAD prefix, same bundled /opt/or-tools --
# so the COPY paths below are unchanged and only the OS package names moved.
#
# Preferred over the alternative fix of giving this flavor its own
# nvidia/cuda:*-ubuntu22.04 base: that would strand EDA tasks on CUDA 12.4 /
# torch 2.6, fork base/gpu's runtime setup across two OS releases, and split
# the tag lineage. Sourcing OpenROAD from noble keeps this a pure overlay on
# the shared base. Also preferred over openroad/ubuntu24.04, which ships a raw
# CMake build tree at /OpenROAD/build with no staged install prefix and no
# openroad on PATH.
#
# If base/gpu's Ubuntu release ever moves again, this image must move with it;
# the ldd guard in the final RUN is what turns that into a build failure
# instead of a task that dies at runtime.
ARG BASE_IMAGE=lbx-tasks-base-gpu
ARG BASE_TAG=runtime-ml-core-py313-local
ARG OPENROAD_PLATFORM=linux/amd64

FROM --platform=${OPENROAD_PLATFORM} openroad/flow-ubuntu24.04-builder:latest AS openroad
FROM ${BASE_IMAGE}:${BASE_TAG}

ENV PATH="/opt/openroad/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/openroad/lib:/opt/or-tools/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64"

# OpenROAD binary tree + its bundled solver libs (or-tools/scip/highs/abseil).
COPY --from=openroad /OpenROAD-flow-scripts/tools/install/OpenROAD /opt/openroad
COPY --from=openroad /opt/or-tools /opt/or-tools

# Runtime system libs OpenROAD's Ubuntu-24.04 binary links against, plus a
# stable CLI symlink. The list is every package owning a resolved non-bundled
# SONAME in `ldd openroad` (dpkg -S inside the source image), minus the ones
# the CUDA base already carries (libc6/libgcc-s1/libstdc++6). Do not hand-edit
# it against a package search: several noble names differ from the obvious
# ones (the t64 time_t transition renamed libglib2.0-0, libpng16-16,
# libpython3.12 and all three Qt5 runtimes).
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libbrotli1 libbsd0 libbz2-1.0 libdouble-conversion3 libexpat1 \
      libfreetype6 libgl1 libglib2.0-0t64 libglvnd0 libglx0 libgomp1 \
      libgraphite2-3 libharfbuzz0b libicu74 libmd0 libmd4c0 libopengl0 \
      libpcre2-16-0 libpcre2-8-0 libpng16-16t64 libpython3.12t64 \
      libqt5charts5 libqt5core5t64 libqt5gui5t64 libqt5widgets5t64 \
      libtcl8.6 libx11-6 libxau6 libxcb1 libxdmcp6 libyaml-cpp0.8 libzstd1 \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /opt/openroad/bin/openroad /usr/local/bin/openroad \
 && ldd /opt/openroad/bin/openroad > /tmp/openroad-ldd.txt \
 && if grep -q "not found" /tmp/openroad-ldd.txt; then \
      echo "unsatisfied OpenROAD SONAMEs -- source image and base OS disagree:" >&2; \
      grep "not found" /tmp/openroad-ldd.txt >&2; exit 1; \
    fi \
 && rm -f /tmp/openroad-ldd.txt \
 && openroad -version

CMD ["/opt/lbx-runtime/.venv/bin/rubric", "mcp"]
