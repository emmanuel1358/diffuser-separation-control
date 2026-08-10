ARG BASE_IMAGE=lbx-tasks-base
ARG BASE_TAG=runtime-ml-core-py313-local
ARG PROBLEM_DIR=examples/xfoil-rust-port

FROM ${BASE_IMAGE}:${BASE_TAG} AS rust-toolchain

ARG PROBLEM_DIR=examples/xfoil-rust-port
ARG TARGETARCH
USER root
COPY ${PROBLEM_DIR}/environment/ /tmp/task-deps/environment/
COPY ${PROBLEM_DIR}/scorer/requirements.txt \
    /tmp/task-deps/scorer/requirements.txt
COPY --chmod=0755 ${PROBLEM_DIR}/environment/install-task-deps.sh \
    /tmp/install-task-deps.sh
RUN test "${TARGETARCH}" = "amd64" \
    && /tmp/install-task-deps.sh /tmp/task-deps \
    && rm -rf /tmp/task-deps /tmp/install-task-deps.sh \
    && test "$(dpkg-query -W -f='${Version}' cargo)" = "1.85.0+dfsg3-1" \
    && test "$(dpkg-query -W -f='${Version}' rustc)" = "1.85.0+dfsg3-1" \
    && test "$(dpkg-query -W -f='${Version}' strace)" = "6.13+ds-1" \
    && strace --version | grep -F "strace -- version 6.13"

FROM rust-toolchain AS cargo-vendor

ARG PROBLEM_DIR=examples/xfoil-rust-port
ENV CARGO_HOME=/opt/cargo-home

# Resolve only from the checked-in lock and manifests. Dummy targets keep this
# layer independent of the solution implementation, so source changes do not
# invalidate the expensive dependency download.
COPY ${PROBLEM_DIR}/solution/reference/Cargo.toml /tmp/cargo-vendor/Cargo.toml
COPY ${PROBLEM_DIR}/solution/reference/Cargo.lock /tmp/cargo-vendor/Cargo.lock
COPY ${PROBLEM_DIR}/solution/reference/crates/rustfoil-core/Cargo.toml \
    /tmp/cargo-vendor/crates/rustfoil-core/Cargo.toml
COPY ${PROBLEM_DIR}/solution/reference/crates/rustfoil-bl/Cargo.toml \
    /tmp/cargo-vendor/crates/rustfoil-bl/Cargo.toml
COPY ${PROBLEM_DIR}/solution/reference/crates/rustfoil-inviscid/Cargo.toml \
    /tmp/cargo-vendor/crates/rustfoil-inviscid/Cargo.toml
COPY ${PROBLEM_DIR}/solution/reference/crates/rustfoil-testkit/Cargo.toml \
    /tmp/cargo-vendor/crates/rustfoil-testkit/Cargo.toml
RUN mkdir -p /opt/cargo-home \
    && for crate in rustfoil-core rustfoil-bl rustfoil-inviscid rustfoil-testkit; do \
        mkdir -p "/tmp/cargo-vendor/crates/${crate}/src"; \
        printf '%s\n' '// dependency-vendoring placeholder' \
            > "/tmp/cargo-vendor/crates/${crate}/src/lib.rs"; \
    done \
    && cargo vendor \
        --manifest-path /tmp/cargo-vendor/Cargo.toml \
        --locked \
        --versioned-dirs \
        /opt/cargo-vendor \
        > /opt/cargo-home/config.toml \
    && rm -rf /tmp/cargo-vendor \
    && chown -R root:root /opt/cargo-home /opt/cargo-vendor \
    && find /opt/cargo-home /opt/cargo-vendor -type d -exec chmod 0555 {} + \
    && find /opt/cargo-home /opt/cargo-vendor -type f -exec chmod 0444 {} +

FROM ${BASE_IMAGE}:${BASE_TAG} AS xfoil-build

ARG PROBLEM_DIR=examples/xfoil-rust-port
USER root
WORKDIR /tmp/xfoil-archive

# The archive is deterministic and checksum-pinned in the task. Build the
# public differential executable from the verified bytes without network I/O.
COPY ${PROBLEM_DIR}/data/xfoil-source.tar.gz \
    ${PROBLEM_DIR}/data/xfoil-source.tar.gz.sha256 ./
RUN sha256sum -c xfoil-source.tar.gz.sha256 \
    && tar -xzf xfoil-source.tar.gz -C /opt \
    && test -f /opt/xfoil-source/src/gpl.txt \
    && cmake -S /opt/xfoil-source -B /tmp/xfoil-build \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/xfoil-build --target xfoil --parallel 2 \
    && install -D -m 0755 \
        /tmp/xfoil-build/src/xfoil-6.97 /opt/xfoil/bin/xfoil \
    && chmod -R a+rX /opt/xfoil-source

FROM rust-toolchain

ARG PROBLEM_DIR=examples/xfoil-rust-port

ENV CARGO_HOME=/opt/cargo-home
ENV CARGO_NET_OFFLINE=true

USER root
COPY --from=cargo-vendor /opt/cargo-home/ /opt/cargo-home/
COPY --from=cargo-vendor /opt/cargo-vendor/ /opt/cargo-vendor/
COPY --from=xfoil-build /opt/xfoil/bin/xfoil /usr/local/bin/xfoil
COPY --from=xfoil-build /opt/xfoil-source/ /data/xfoil-source/
COPY --chown=root:root \
    ${PROBLEM_DIR}/data/transform_runner.py \
    ${PROBLEM_DIR}/data/public_cases.json \
    ${PROBLEM_DIR}/data/XFOIL_PROVENANCE.md \
    /data/
RUN test -x /usr/local/bin/xfoil \
    && test -r /data/xfoil-source/src/gpl.txt \
    && su -s /bin/sh agent -c 'test -r /opt/cargo-home/config.toml' \
    && ! su -s /bin/sh agent -c 'test -w /opt/cargo-home' \
    && ! su -s /bin/sh agent -c 'test -w /opt/cargo-vendor'

WORKDIR /workdir
RUN mkdir -p /mcp_server/data /mcp_server/grader /workdir /tmp/output/repo \
    && chown -R 1000:1000 /workdir /tmp/output

COPY --chown=1000:1000 ${PROBLEM_DIR}/starter/ /tmp/output/repo/
RUN su -s /bin/sh agent -c \
        'cd /tmp/output/repo && cargo build --release --offline \
            --bin transform-candidate' \
    && rm -rf /tmp/output/repo/target /tmp/output/repo/Cargo.lock \
    && chmod 0755 /data/transform_runner.py

COPY --chown=root:root ${PROBLEM_DIR}/scorer/data/ /mcp_server/data/
COPY --chown=root:root ${PROBLEM_DIR}/scorer/ /mcp_server/grader/
RUN rm -rf /mcp_server/grader/data \
    && chown -R root:root /mcp_server/data /mcp_server/grader \
    && chmod 0700 /mcp_server \
    && find /mcp_server/data /mcp_server/grader -type d -exec chmod 0700 {} + \
    && find /mcp_server/data /mcp_server/grader -type f -exec chmod 0600 {} + \
    && ! su -s /bin/sh agent -c \
        'test -r /mcp_server/data/hidden_requests.json' \
    && ! su -s /bin/sh agent -c 'test -r /mcp_server/grader/compute_score.py' \
    && ! su -s /bin/sh agent -c \
        'test -r /mcp_server/grader/candidate_monitor.py' \
    && test "$(stat -c '%U:%G:%a' \
        /mcp_server/grader/candidate_monitor.py)" = "root:root:600" \
    && test ! -e /solution \
    && test ! -e /mcp_server/reference \
    && test ! -e /tmp/flexfoil-reference \
    && test ! -e /opt/flexfoil-reference \
    && test -z "$(find /data /opt/cargo-home /opt/cargo-vendor \
        /tmp/output/repo -name ceiling_responses.json -print -quit)" \
    && test ! -e /tmp/output/repo/.alignerr

COPY ${PROBLEM_DIR}/task.toml ${PROBLEM_DIR}/instruction.md /task/
