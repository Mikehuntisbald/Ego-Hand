# Supply an immutable, locally audited Python/CUDA base. No network, git fetch,
# source injection or unpinned pip resolution is permitted during this build.
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS dependencies
USER root
WORKDIR /opt/ego-hand
COPY wheelhouse/ /opt/wheelhouse/
COPY wheelhouse_sha256.json /opt/wheelhouse_sha256.json
COPY extra_wheels/ /opt/extra_wheels/
COPY extra_wheels_sha256.json /opt/extra_wheels_sha256.json
COPY tools/verify_runtime_release.py /opt/ego-hand/tools/verify_runtime_release.py
RUN python3 tools/verify_runtime_release.py --manifest /opt/wheelhouse_sha256.json --root /opt/wheelhouse \
 && if ls /opt/extra_wheels/*.whl >/dev/null 2>&1; then python3 tools/verify_runtime_release.py --manifest /opt/extra_wheels_sha256.json --root /opt/extra_wheels; fi \
 && python3 -m venv --without-pip --system-site-packages /opt/ego-runtime-env \
 && /opt/ego-runtime-env/bin/python -m pip install --ignore-installed --no-index --no-deps /opt/wheelhouse/*.whl \
 && if ls /opt/extra_wheels/*.whl >/dev/null 2>&1; then /opt/ego-runtime-env/bin/python -m pip install --ignore-installed --no-index --no-deps /opt/extra_wheels/*.whl; fi \
 && if [ -f /opt/wheelhouse/chumpy-0.70.tar.gz ]; then /opt/ego-runtime-env/bin/python -m pip install --ignore-installed --no-index --no-deps --no-build-isolation /opt/wheelhouse/chumpy-0.70.tar.gz; fi \
 && sed -i 's/include-system-site-packages = true/include-system-site-packages = false/' /opt/ego-runtime-env/pyvenv.cfg \
 && rm -rf /opt/wheelhouse /opt/extra_wheels
FROM ${BASE_IMAGE}
USER root
WORKDIR /opt/ego-hand
COPY --from=dependencies /opt/ego-runtime-env /opt/ego-runtime-env
COPY ego_hand_runtime/ /opt/ego-hand/ego_hand_runtime/
COPY configs/runtime/ /opt/ego-hand/configs/runtime/
COPY tools/ /opt/ego-hand/tools/
COPY vendor/ /opt/ego-hand/vendor/
COPY tests/runtime/ /opt/ego-hand/tests/runtime/
COPY research/hand3d_rollout_v8.py /opt/ego-hand/research/hand3d_rollout_v8.py
COPY research/native_mano_h20_20261019/v2/src/native_ba.py /opt/ego-hand/research/native_mano_h20_20261019/v2/src/native_ba.py
COPY runtime_source_sha256.json /opt/ego-hand/runtime_source_sha256.json
RUN /opt/ego-runtime-env/bin/python tools/verify_runtime_release.py --manifest /opt/ego-hand/runtime_source_sha256.json
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 VIRTUAL_ENV=/opt/ego-runtime-env
LABEL org.ego-hand.scope="Independent runtime candidate; acceptance is determined by versioned execution receipts"
ENTRYPOINT ["/opt/ego-runtime-env/bin/python", "/opt/ego-hand/tools/verified_runtime_entry.py"]
CMD ["--help"]
