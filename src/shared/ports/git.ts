// Git: one seam between rig and the machine. A port names a capability, never a tool.
export interface Git {
  revParse(dir: string, ref: string): Promise<string | null>;
  /** no tracked file in dir differs from HEAD and no untracked one sits in it (ignored files aside):
   *  the engine's CMake globs its CUDA sources (ggml-cuda/*.cu, template-instances/mmq*.cu), so an
   *  untracked source is compiled into a build stamped with the pin's sha */
  isClean(dir: string): Promise<boolean>;
  /** clone one commit of a repo into dir (init + fetch --depth 1 + checkout) */
  fetchCommit(dir: string, repo: string, sha: string): Promise<void>;
}
