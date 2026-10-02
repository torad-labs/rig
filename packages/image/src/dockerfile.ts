// The Dockerfile an image is built from, rendered from data: a driver-only Ubuntu 22.04 (the oldest
// glibc a prebuilt supports) holding the rig CLI, the head, the engine pin and the engine for one
// sm, installed by `rig build` exactly as a machine installs it. Nothing NVIDIA ships is kept but the runtime rig installs beside
// the engine: the container toolkit hands a container the host's driver at run time. What the install step only reads (the
// engine tarball, the compat libcuda) is bound into that step, never copied: a COPY is a layer every pull carries even after
// a later step deletes its files, 375 MB of the 1.26 GB image of 2a9d696 (321 MB compat, 54 MB tarball; docker history).

/** base images, each pinned by digest: a tag moves, a digest does not */
export const IMAGES = {
  base: "ubuntu:22.04@sha256:b8b6ee6aa931ecd9d0d952abc34dc0e5f7c6a30c6bb71b079fe399fde0329c02",
  /** only its driver-compat libcuda, for the one build step that checks every symbol resolves */
  cudaCompat:
    "nvidia/cuda:13.0.0-base-ubuntu22.04@sha256:c08986c5ea16a41bba86f4ac08b327861c7b7d05dee7fae0648dcf0e964bf6e4",
} as const;

/** where rig lives in the image: its root, so the binary finds heads/ and engine/ above dist/ */
export const IMAGE_ROOT = "/opt/rig";

/** a rented box is reached over ssh only (the server binds loopback): a rental's SSH launch mode
 *  runs the image's sshd */
const SSHD = "openssh-server";

export interface DockerfileInputs {
  head: string;
  cap: string;
  /** the rig commit the source is */
  commit: string;
  /** the engine tarball in the context's engine/, or none when the pin publishes a prebuilt for the cap */
  engineTarball?: string;
  /** what the image installs with apt: what prepare asks of a machine whose engine is installed */
  packages: readonly string[];
}

export function renderDockerfile(inputs: DockerfileInputs): string {
  const { head, cap, commit } = inputs;
  const tarball = inputs.engineTarball;
  const from = tarball ? ` --from-tarball /tmp/rig-engine/${tarball}` : "";
  const engine = tarball ? `--mount=type=bind,source=engine,target=/tmp/rig-engine ` : "";
  return `# ${head} on sm_${cap}, rig ${commit}: rendered by \`rig image ${head}\`, never edited by hand
FROM ${IMAGES.cudaCompat} AS compat

FROM ${IMAGES.base}
RUN apt-get update -qq \\
 && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends ${[...inputs.packages, SSHD].join(" ")} \\
 && rm -rf /var/lib/apt/lists/*
COPY rig/ ${IMAGE_ROOT}/
# rig build checks every symbol the engine imports resolves (ldd -r), libcuda.so.1's among them: the
# driver's is not here until a container runs, so the compat libcuda stands in for this step only,
# bound like the engine tarball and gone with the step
RUN --mount=type=bind,from=compat,source=/usr/local/cuda/compat,target=/tmp/rig-compat ${engine}\\
    echo /tmp/rig-compat > /etc/ld.so.conf.d/zz-rig-build.conf && ldconfig \\
 && ${IMAGE_ROOT}/dist/rig build ${head} --cap ${cap}${from} \\
 && rm -rf /etc/ld.so.conf.d/zz-rig-build.conf ${IMAGE_ROOT}/local/downloads \\
 && ldconfig
ENV PATH=${IMAGE_ROOT}/dist:$PATH \\
    NVIDIA_VISIBLE_DEVICES=all \\
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
CMD ["rig", "up", "${head}", "--foreground"]
`;
}
