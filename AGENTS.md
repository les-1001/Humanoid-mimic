## Humanoid Mimic project rules

### Project summary
- Humanoid Mimic is a public portfolio website for ML + robotics internships.
- A visitor uses a laptop webcam; a 3D Unitree G1 copies their movements in real time.
- Engineering rigour, honest evaluation and a reliable demo take priority over flashy features.
- LIVE MODE runs entirely in the browser, with no server.
- Live pipeline: webcam -> person detector -> small student pose network -> temporal smoothing -> upper-body IK -> G1 rendering.
- Student inference uses ONNX Runtime Web, WebGPU EP, with WASM fallback.
- Simulation uses MuJoCo WebAssembly (`@mujoco/mujoco`); rendering uses three.js; target is 20-30 fps on an ordinary laptop.
- Default UPPER-BODY mode supports seated visitors; optional FULL-BODY mode requires standing 2-3 m back.
- RECORD MODE sends a 10 s recording to a Hugging Face ZeroGPU Gradio Space with a server GPU.
- Record pipeline: large teacher network -> full-body SMPL-X motion -> GMR retargeting to `unitree_g1` (29 DoF) -> MuJoCo physics feasibility check -> smooth replay + feasibility score.
- Train the teacher (video -> SMPL-X pose/shape/camera, with a temporal module) on BEDLAM 2.0 using NUS SoC SLURM A100 40GB GPUs.
- Train a small student distilled from the teacher and quantised for browser inference.
- Off the shelf: GMR (MIT), MuJoCo Menagerie G1, a person detector and three.js. All robot work is simulated; there is no robot hardware.
- The developer is an NUS CS undergraduate with PyTorch + Ultralytics YOLO experience, using a Windows laptop and WSL2 Ubuntu 24.04; this repo lives in WSL.
- Split train/val/test BY SEQUENCE, never by frame. SSH terminal choice is unresolved; confirm WSL `ssh soc` versus PowerShell ssh/scp with `\\wsl$` paths before providing transfer commands.

### Session start
Read AGENTS.md, .specify/memory/constitution.md, docs/PROGRESS.md, docs/decisions.md and docs/cluster.md before doing anything. If working on a feature, also read its specs/NNN-*/ folder.

### TOOL ROUTING
- Spec Kit owns WHAT and the PLAN: specs/NNN-*/spec.md, plan.md, research.md, tasks.md. There is exactly one plan per feature: tasks.md.
- Do NOT use the Superpowers skills brainstorming, writing-plans or executing-plans for work that has (or should have) a specs/NNN-* folder. If a Superpowers skill tries to start a design or plan, skip it and follow tasks.md.
- Do NOT create git worktrees. Spec Kit's per-feature git branch is the isolation.
- During $speckit-implement, use Superpowers for HOW: test-driven-development for every task that adds or changes code (failing test first); systematic-debugging for any failure, including cluster logs I paste; verification-before-completion before marking any task [X] (run the command that proves it and show the output); requesting-code-review at the end of each phase/user story in tasks.md; finishing-a-development-branch only after $speckit-converge reports Converged.

### CLUSTER BOUNDARY
You cannot run anything on the cluster: login needs my password + OTP. Tasks that need the cluster are tagged [USER-RUN] in tasks.md. When you reach one: write all needed files locally, re-read them to confirm their content, give me short single-line commands (one per line), then STOP and wait. Never mark a [USER-RUN] task done without cluster output I pasted that proves it.

### End of every work session
Append to docs/PROGRESS.md the date, feature/task IDs touched, what was done, commands the user must run, open questions and the exact next message the user should send.

### Verification and file integrity
Never claim something works without running it or seeing pasted output. After editing any file a cluster job depends on, re-read it before telling me to upload it.
