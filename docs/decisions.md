# Decisions

**PUBLIC RELEASE BLOCKER — BEDLAM-trained teacher/student deployment and distribution are not explicitly cleared by the published licence; SMPL-X files must not be publicly bundled.** Non-commercial research permission is not an explicit public-service or derived-weight distribution grant. Obtain written permission covering hosted inference, browser-delivered distilled ONNX weights, and third-party hosting. Until then, keep the research pipeline private; a public fallback can use separately licensed, unmodified pose estimation and robot animation, with no BEDLAM/SMPL-X assets. That fallback is a proposed workaround, not implemented feature work. [BEDLAM licence][B2-license] [SMPL-X licence][SX-license]

**DATA COMPLETENESS BLOCKER — videos plus camera CSV/JSON are insufficient for the documented SMPL-X training pipeline.** Also obtain `b2_motions_npz_training.tar` and sequence metadata, including `be_seq.csv`, through registered BEDLAM access; their complete official file list/sizes and canonical identity/motion mapping are not publicly verifiable here. Workaround: inspect the registered manifest and a small authorized scene before purchasing storage or starting training. [Author preprocessing instructions][B2-process] [Author parser][B2-parser]

**RUNTIME MISMATCH — the current project requires Python 3.11, while ZeroGPU currently lists Python 3.10.13 and 3.12.12.** Proposed resolution: a separately pinned Python 3.12.12 Space environment; keep local/cluster dependencies separate. No project configuration was changed in this research. [ZeroGPU compatibility][HF-zero] [Current project configuration](../pyproject.toml)

Research date: **2026-10-28 (Asia/Singapore)**. Entries distinguish primary-source facts, source-code inspection, project choices, and unverified points. Public non-commercial portfolio use is the intended use; being free or open source does not automatically settle dataset/body-model permissions. The evidence files contain observed metadata, not model benchmarks. Numeric lookup keys are `bedlam`, `g1`, `head_download_sizes`, `backbone_files`, `package_metadata`, and `published_observations` in the source audit; per-file BEDLAM sizes are under `files[]` in the manifest: [source audit](research/stack-source-audit-2026-10-28.json), [exact BEDLAM mirror file manifest](research/bedlam2-download-manifest-2026-10-28.json). No private download, model execution, browser benchmark, or cluster job was performed.

| ID | Decision | Options | Choice | Why | Sources | Date |
| --- | --- | --- | --- | --- | --- | --- |
| D-01 | Training data | BEDLAM official / HF mirror | Selected MP4 + metadata archives; gated motion labels separately | Avoid image/depth bulk; permission conditional | [Official][B2-home], [mirror][B2-hf] | 2026-10-28 |
| D-02 | Body model | Public bundle / private runtime / offline | Offline licensed use; private runtime only after written clearance | No redistribution permission | [Licence][SX-license] | 2026-10-28 |
| D-03 | Retargeter | GMR SMPL-X / GVHMR | GMR with explicit SMPL-X adapter | Inspectable contract | [Source][GMR-utils] | 2026-10-28 |
| D-04 | Robot asset | Menagerie / GMR MJCF | Menagerie canonical; map GMR output by name | Limits/controllers differ | [Menagerie][MG-xml], [GMR][GMR-xml] | 2026-10-28 |
| D-05 | Browser physics | Official MuJoCo WASM / forks | Official `@mujoco/mujoco@3.9.0` baseline | Required API is present in published package | [Package][MJ-npm] | 2026-10-28 |
| D-06 | Browser inference | FP32 / FP16 / INT8 | ORT Web; FP32 fallback, conditional FP16 and INT8 | Hardware/operator support varies | [Web guide][ORT-web] | 2026-10-28 |
| D-07 | Primary real-world test | EMDB / 3DPW / RICH | 3DPW official test, offline only | Published protocol and verified manageable download size | [Protocol][PW-eval] | 2026-10-28 |
| D-08 | Person detector | Ultralytics / MediaPipe / RTMDet | RTMDet-tiny, ONNX export | Apache release; one inference runtime | [Model zoo][RTM-det], [licence][RTM-license] | 2026-10-28 |
| D-09 | Teacher/student backbones | DINO/HMR/ViTPose; MobileNet/EfficientViT/RepViT | DINOv2 ViT-B/14 + MobileNetV4 Conv Small | Explicit Apache-2.0 weights | [DINOv2][DINO2], [MobileNet card][MB4] | 2026-10-28 |
| D-10 | Hosting | Static / ZeroGPU / paid GPU | Static LIVE + Gradio ZeroGPU RECORD, conditional release | Free tiers exist; quotas need fallback | [ZeroGPU][HF-zero], [static][HF-static] | 2026-10-28 |
| D-00 | Overall stack | See individual entries | Stack below; release gates remain closed | Research choice, not permission or performance claim | [D-00](#d-00-overall-stack) | 2026-10-28 |

## D-01 BEDLAM 2.0

**Decision / Options / Choice / Why.** Choose the official HF mirror for selected video/camera archives after accepting its gate; use the registered official site for missing motion labels and metadata. Do not download PNG, EXR depth, render assets or overview plots. This is a project choice based on the separate mirror archive layout, not an assertion that all training inputs are mirrored. [Mirror tree][B2-tree] [Preprocessing][B2-process]

**Licence and public demo.** The custom BEDLAM2 licence permits personal non-commercial scientific research, education and artistic projects; it prohibits redistribution and specified commercial, surveillance, military and other uses, including training for prohibited purposes. It does not explicitly answer whether this internship portfolio may publish trained or distilled weights or run unrestricted public inference. **Unverified: permission for those exact uses.** Ask the rights holder, describing both the server teacher and downloadable browser student; do not infer approval from other authors' demos. [Licence][B2-license] [Official contact][B2-home]

**Official site versus HF.** The official announcement dated 2026-02-26 concerns the image/depth mirror. The current official-linked `Intelligent-Systems/BEDLAM2` mirror also lists MP4 and camera/scene GT; depth has a separate mirror. Both routes require licence acceptance; the official download page redirects to login without registration. The official headline sizes are MP4 160 GB, PNG 11 TB and depth 15 TB; these are publisher approximations, not our storage measurements. [Announcement/download description][B2-home] [Mirror card][B2-hf] [Registered download][B2-download]

**Exact public file inventory and sizes.** The public tree API was paginated to exhaustion. At revision `15e18597eb570e1d7166a0b9ef1009d35b904093`, the selected mirror subset contains:

| Group | Archives/files | Exact bytes |
| --- | ---: | ---: |
| MP4 `.tar` archives | 66 | 170,586,306,560 |
| Non-CSV metadata `.tar.gz` archives | 66 | 2,626,676,635 |
| CSV metadata `.tar.gz` archives | 66 | 516,632,850 |
| The above archives only | 198 | 173,729,616,045 |
| Above + checksum sidecars + available READMEs/blacklists + licence | 414 | 173,729,646,373 |

Source and calculation: [HF public tree API][B2-api], [complete selected file manifest](research/bedlam2-download-manifest-2026-10-28.json), [audit](research/stack-source-audit-2026-10-28.json). Byte counts are compressed archive/download sizes. They exclude gated motion/body-model files, decoded frames, caches and working space. The API masks content hashes without access; no archive integrity or interior contents were verified. Non-CSV GT is advertised as JSON/scene metadata, but its actual internal schema remains uninspected. [Mirror card][B2-hf] [API][B2-api]

For each prefix `S` in the table, exact archive paths are `S/mp4/S_mp4.tar`, `S/ground_truth/S_gt_centersubframe_exr_meta.tar.gz`, and `S/ground_truth/S_gt_centersubframe_exr_meta_csv.tar.gz`. Each has a `.xxh128` sidecar. Despite `exr_meta` in the name, these are listed under ground truth, not the separate depth dataset. Full paths, byte sizes, sidecars and small metadata files are in the JSON manifest. [HF tree][B2-tree]

| Scene/archive prefix `S` | MP4 bytes | Non-CSV metadata bytes | CSV metadata bytes |
| --- | ---: | ---: | ---: |
| `20240416_1_171_yogastudio_orbit_timeofday` | 2,405,939,200 | 34,182,708 | 7,878,333 |
| `20240423_1_171_yogastudio_staticloc_timeofday` | 1,815,531,520 | 19,196,897 | 3,012,363 |
| `20240424_1_171_citysample_orbit` | 4,769,341,440 | 34,079,199 | 7,744,738 |
| `20240425_1_171_citysample_dolly` | 3,658,557,440 | 32,411,190 | 7,358,680 |
| `20240425_1_171_hdri` | 681,861,120 | 17,132,190 | 2,919,079 |
| `20240426_5_100_citysample_orbit` | 1,788,416,000 | 12,509,820 | 2,870,037 |
| `20240429_1_171_stadium` | 1,250,508,800 | 17,343,689 | 2,915,218 |
| `20240502_5_200_citysample_dolly` | 2,816,491,520 | 24,226,319 | 5,558,499 |
| `20240506_10_200_hdri` | 906,076,160 | 21,030,933 | 5,130,186 |
| `20240506_5_200_citysample_orbit` | 3,517,685,760 | 24,903,392 | 5,714,516 |
| `20240507_5_200_citysample_dollyz` | 2,052,874,240 | 24,477,860 | 5,612,829 |
| `20240514_1_1001_citysample_tracking` | 8,702,945,280 | 64,933,413 | 14,785,516 |
| `20240604_5_500_citysample_tracking` | 2,478,417,920 | 18,884,039 | 4,383,943 |
| `20240605_3_500_busstation_tracking` | 1,571,461,120 | 21,402,928 | 4,970,018 |
| `20240606_1_500_stadium_closeup` | 1,450,813,440 | 16,973,923 | 2,985,944 |
| `20240606_4_250_busstation_orbit` | 861,982,720 | 10,057,728 | 2,363,802 |
| `20240611_5_200_citysamplenight_dolly` | 800,604,160 | 7,564,597 | 1,785,395 |
| `20240611_5_250_archmodelsvol8_dolly` | 1,896,325,120 | 9,249,767 | 2,194,035 |
| `20240613_1_200_hdri` | 476,477,440 | 11,717,031 | 2,831,851 |
| `20240614_1_300_hdri` | 944,885,760 | 10,422,917 | 1,842,854 |
| `20240614_5_200_citysamplenight_tracking` | 839,219,200 | 7,603,459 | 1,769,517 |
| `20240617_10_500_hdri` | 840,519,680 | 16,475,121 | 4,168,899 |
| `20240618_1_500_ai0805_orbit` | 4,625,029,120 | 30,613,122 | 7,139,492 |
| `20240619_1_250_ai1004_tracking` | 1,280,020,480 | 16,876,827 | 3,891,353 |
| `20240619_2_250_ai1004_orbit` | 1,037,045,760 | 11,969,214 | 2,803,877 |
| `20240620_5_250_archmodelsvol8_dollyz` | 1,098,342,400 | 9,724,068 | 2,287,487 |
| `20240625_1_2337_hdri` | 4,494,807,040 | 96,937,975 | 17,314,665 |
| `20240628_1_250_ai1004_tracking` | 1,431,982,080 | 19,847,594 | 4,565,124 |
| `20240628_4_250_busstation_orbit` | 889,149,440 | 11,111,172 | 2,622,664 |
| `20240701_1_250_ai0901_lookat` | 2,101,360,640 | 14,263,172 | 2,701,526 |
| `20240703_1_250_ai0901_orbit_portrait` | 2,963,517,440 | 20,400,429 | 4,674,440 |
| `20240708_1_250_ai0901_static_portrait` | 1,363,630,080 | 6,453,332 | 401,515 |
| `20240709_5_250_archmodelsvol8_zoom` | 2,152,478,720 | 7,443,914 | 1,724,039 |
| `20240710_1_250_ai0805_orbit_portrait` | 2,781,265,920 | 19,293,580 | 4,414,840 |
| `20240711_5-10_250_busstation_orbit_zoom` | 856,637,440 | 11,393,630 | 2,915,712 |
| `20240725_1_250_ai0805_vcam` | 3,784,704,000 | 26,972,663 | 10,981,068 |
| `20240726_1_250_ai0805_vcam` | 3,911,075,840 | 26,968,703 | 10,982,223 |
| `20240729_1_250_ai1004_vcam` | 1,853,603,840 | 26,664,747 | 10,728,691 |
| `20240730_1_250_ai1101_vcam` | 1,577,308,160 | 26,142,311 | 10,489,500 |
| `20240731_1_1827_hdri` | 5,243,310,080 | 74,075,826 | 13,183,545 |
| `20240806_1_250_ai1101_vcam` | 1,578,792,960 | 25,643,169 | 10,302,661 |
| `20240808_1_250_ai1105_vcam` | 2,096,977,920 | 26,373,725 | 10,738,000 |
| `20240809_1_250_ai1102_vcam` | 2,082,426,880 | 26,733,923 | 10,746,358 |
| `20240813_1_250_ai1004_tracking` | 1,404,026,880 | 22,470,974 | 7,769,249 |
| `20241001_5-10_250_busstation_orbit_zoom` | 846,960,640 | 21,823,997 | 3,639,695 |
| `20241107_1_250_archmodelsvol8_tracking` | 3,035,668,480 | 43,457,876 | 3,911,776 |
| `20241114_1_4619_hdri` | 13,385,779,200 | 498,387,109 | 78,611,450 |
| `20241204_1_2120_hdri` | 5,675,857,920 | 212,561,818 | 33,584,222 |
| `20241210_5-10_250_rome_dollyz_zoom` | 2,226,503,680 | 30,270,530 | 5,427,721 |
| `20241211_5-10_250_rome_orbit_zoom` | 1,945,835,520 | 30,678,502 | 5,527,603 |
| `20241212_5-10_250_rome_dolly_zoom` | 1,724,180,480 | 30,560,612 | 5,497,278 |
| `20241213_1_250_rome_tracking` | 4,054,702,080 | 59,540,075 | 10,764,856 |
| `20241217_1_250_rome_vcam` | 4,126,218,240 | 58,402,655 | 10,629,298 |
| `20241219_5_250_chemicalplant_dollyz_zoom` | 1,611,120,640 | 32,183,109 | 5,749,343 |
| `20250103_1_250_rome_vcam` | 6,033,653,760 | 66,446,061 | 12,060,700 |
| `20250110_1_250_chemicalplant_vcam` | 3,323,422,720 | 62,894,877 | 11,533,242 |
| `20250113_1_250_rome_vcam` | 5,278,720,000 | 64,803,290 | 11,874,039 |
| `20250114_4-5_250_chemicalplant_dolly_zoom` | 1,654,937,600 | 31,831,187 | 5,696,991 |
| `20250123_1_250_chemicalplant_vcamego` | 4,540,026,880 | 58,934,132 | 8,979,212 |
| `20250131_1_250_ai1102_vcamego` | 3,236,106,240 | 68,155,820 | 10,330,209 |
| `20250206_4-7_250_yakohama_vcamego_approach` | 2,527,549,440 | 70,412,159 | 10,656,264 |
| `20250211_1_250_ai1105_upperbody` | 1,891,133,440 | 29,197,876 | 4,655,825 |
| `20250212_1_250_yakohama_upperbody` | 827,279,360 | 27,984,675 | 4,463,645 |
| `20250214_1_250_chemicalplant_upperbody` | 3,070,924,800 | 35,416,501 | 5,681,921 |
| `20250218_2-3_250_middleeast_upperbody` | 1,704,550,400 | 47,543,189 | 7,618,862 |
| `20250219_3-4_250_middleeast_vcam_approach` | 730,746,880 | 30,037,395 | 5,564,417 |

**Download method.** After official registration and HF licence-gate approval, use `hf download` or `snapshot_download` with the pinned revision and an explicit filename allowlist from the manifest. HF supports filename/pattern selection. Yes, a subset can be selected per scene/archive; independent MP4 files *inside* a tar cannot be assumed individually addressable. Do not call an unrestricted snapshot download. Decode only needed frames in job-local `/tmp`, keeping scratch in shards. These are proposed operating choices; no download was executed. [HF selective-download documentation][HF-download] [Archive listing][B2-tree] [Cluster rules](cluster.md)

**Additional required inputs, not publicly size-verified:**

| Input | Verified role | Size/access limit |
| --- | --- | --- |
| `b2_motions_npz_training.tar` | Required motion NPZ archive in author training instructions; contains the referenced SMPL-X motion data | Exact bytes and per-motion selective-download availability unverified behind official registration |
| Per-scene `be_seq.csv` | Parser reads `Comment`, `Body`, `X`, `Y`, `Z`, `Yaw` | Packaging/location in registered download manifest and byte sizes unverified |
| `smplx_lockedhead_20230207.zip` | Author downloader selects locked-head SMPL-X, with processing configured for 16 betas | Separate SMPL-X registration; exact bytes unverified |
| `checkpoints/data_utils/utils.zip` | Listed by author downloader as processing utilities | Exact bytes unverified; obtain only needed licensed contents |

Sources for every row: [preprocessing README][B2-process], [parser][B2-parser], [download script][B2-download-script]. The official source pipeline expects PNG input; an MP4-decoding adapter will be future work and must preserve frame numbering and camera alignment. Compression makes MP4-derived images a different input protocol from lossless PNG; disclose that choice in comparisons. [Parser][B2-parser] [Official media formats][B2-home]

**Leakage-free split metadata.** Verified from the author parser, not guessed from archive names:

| Split/association field | Available evidence | Remaining limitation |
| --- | --- | --- |
| Scene/render batch | `bedlam2_scene_names.csv`: `Folder name`, `Scene name` | Scene name alone is not a unique person/motion identity |
| Sequence | `be_seq.csv` `Comment` contains `sequence_name=...`; used to locate the frame folder and camera file | Preserve `(scene folder, sequence_name)` together |
| Body/animation asset | `Body` selects `<Body>.npz`; `Comment` supplies `start_frame` | This identifies an asset/trajectory reference; it is not proven to be a canonical body-shape ID or original motion-clip ID |
| Motion and shape | Motion NPZ reads `poses`, `trans`, `betas`, `gender` | A canonical source-motion ID, overlap interval and reusable body identity must be inspected after registration; exact additional keys unverified |
| Camera | `ground_truth/meta_exr_csv/<sequence_name>_camera.csv`: `x`, `y`, `z`, `yaw`, `pitch`, `roll`, `focal_length` | Intrinsics/extrinsics are reconstructed; do not assume a separate persistent camera-ID key |

Sources: [scene index][B2-scenes], [parser][B2-parser]. The BEDLAM2 writer saves `pose_cam`, `pose_world`, `shape`, `trans_cam`, `trans_world`, `cam_int`, `cam_ext` and other targets, but does **not** preserve the original `Body`/motion association in its final save list; retain provenance before conversion. Some rendered subjects have withheld training labels and are skipped by the author code. [Parser][B2-parser] [README caveat][B2-process]

**Project split choice:** build connected groups sharing sequence, canonical body identity or original motion clip; never split reused motions or alternate cameras across train/val/test. Record missing metadata and block an identity/clip-disjoint claim until the mappings are verified. This follows the [constitution](../.specify/memory/constitution.md); it is not a claim that the public CSV already supplies all mappings.

**Sources / Date:** primary links above; 2026-10-28.

## D-02 SMPL-X body model

**Decision / Options / Choice / Why.** Reject public bundling. Use the personally licensed body model offline for research; allow a private server runtime only after written clearance for the proposed public demo and hosting arrangement. [Model licence][SX-license]

**Licence.** The grant is personal, single-user and non-transferable for non-commercial research, education or artistic use; it prohibits copying/sharing/distribution and making materials available to third parties without prior written permission. A public HF repository containing `.pkl`/`.npz` body files is not allowed under that grant. **Unverified: a public inference service using private files is permitted for this portfolio.** [Model licence, grant/no-distribution clauses][SX-license]

**Registration, files and download.** You must register and accept the SMPL-X terms yourself. GMR documents `assets/body_models/smplx/SMPLX_NEUTRAL.pkl`, `SMPLX_FEMALE.pkl`, `SMPLX_MALE.pkl`; only the selected gender is used for a particular call, and GVHMR uses neutral. BEDLAM2 instead documents the locked-head/no-bun release `smplx_lockedhead_20230207.zip` and 16 shape components, with `.npz` models under `models_lockedhead/smplx`. Do not silently substitute the ordinary model or truncate betas. Exact archive/file byte sizes cannot be verified before registered access. [Registration/download][SX-download] [GMR model layout][GMR-readme] [Loader][GMR-utils] [BEDLAM downloader][B2-download-script] [BEDLAM FAQ][B2-faq]

**Conditional private-runtime design.** If written permission covers it, store the licensed files in an owner-controlled private HF repository; use a fine-grained read token limited to that repository, saved as a server-side Space Secret, and download a pinned revision during startup. Exclude files from Git, public build layers, Gradio output directories, logs and client responses. This is an access-control design, **not a licence workaround**: private cloud copies and public service use still require clearance. Prefer an authorized institutional host if the rights holder disallows HF storage. [HF private downloads][HF-download] [Space secrets][HF-overview] [Token scopes][HF-tokens] [SMPL-X restrictions][SX-license]

**Sources / Date:** above; 2026-10-28.

## D-03 GMR and motion interchange

**Decision / Options / Choice / Why.** Choose GMR with a project-owned explicit adapter to its SMPL-X contract; retain GVHMR ingestion as a documented alternative, not a second teacher. GMR is MIT: public use/modification/distribution is allowed with its notice; that licence does not grant rights to SMPL-X models or upstream motion data. Download code at inspected revision `25f56db79ac411258b377b6d4ccba0c8ff3372ac`; the G1 asset subtree has 69 blobs totalling 54,380,454 bytes (both robot variants and assets, not a compressed repo download). [MIT][GMR-license] [Tree API][GMR-tree] [Model preparation][GMR-readme]

**Robot.** `unitree_g1` selects `assets/unitree_g1/g1_mocap_29dof.xml`, not the with-hands variant. It has 29 articulated hinge joints: each leg 6, waist 3, each arm 7; the floating base is additional state. Names, ranges and torque metadata are tabulated in D-04. [Robot mapping][GMR-params] [MJCF][GMR-xml]

**Exact SMPL-X file contract consumed by `scripts/smplx_to_robot.py`:** NumPy `.npz`, loaded with `np.load(..., allow_pickle=True)`; axis-angle rotations are radians and translations use the body model's metric convention. Inputs are a trajectory of N frames, not a dictionary of rendered vertices. [Script][GMR-script] [Loader and rotation conversion][GMR-utils]

| Key | Shape/type consumed | Meaning |
| --- | --- | --- |
| `gender` | Scalar convertible with `str(...)` to `neutral`, `male` or `female` | Selects SMPL-X body model |
| `pose_body` | `(N, 63)` numeric | 21 body joints, axis-angle |
| `root_orient` | `(N, 3)` numeric | Global root axis-angle |
| `trans` | `(N, 3)` numeric | Root translation |
| `betas` | One shape vector, flattened to `(1, B)` | Shared body shape; source comment expects 16, but loader does not explicitly pass `num_betas=16` |
| `mocap_frame_rate` | NumPy scalar with `.item()` | Source fps, exact spelling required |

Source for every field: [pinned `smpl.py`][GMR-utils]. The loader sets `use_pca=False`, zeros hands/jaw/eyes, requests full pose, and estimates height from the first beta. Verify the selected model's beta dimension explicitly; compatibility with BEDLAM's locked-head model is **not runtime-tested**. Its offline converter produces named joint global positions and `wxyz` quaternions. [Loader][GMR-utils]

**FPS and script gotchas.** Target fps is hardcoded to 30. Downsampling uses an integer `int(src_fps / tgt_fps)` step count with interpolation; lower/equal source fps is labelled with the target rate without a general resampler. The main loop increments its index before reading, skipping frame zero in non-loop mode. Treat these as source-inspected adapter hazards, not fixes performed here. [SMPL-X script][GMR-script] [Resampling code][GMR-utils]

**Saved pickle dictionary:** `fps` is `aligned_fps`; `root_pos` is `(T,3)`; `root_rot` is `(T,4)` **xyzw**, explicitly reordered from MuJoCo wxyz; `dof_pos` is `(T,29)` for this G1; `local_body_pos=None`; `link_body_list=None`. Here T is the number of frames actually saved, not automatically the input N. A pickle is a Python interchange artifact, not a browser-ready file or a physics-feasibility certificate. [Save block][GMR-script]

**Alternative `hmr4d_results.pt`.** `load_gvhmr_pred_file` uses `torch.load` and consumes only the nested `smpl_params_global` dictionary: `body_pose` `(N,63)`, `global_orient` `(N,3)`, `transl` `(N,3)`, and `betas` normally `(N,10)`. It takes `betas[0]`, pads six coefficients to 16, converts tensors with `.numpy()`, chooses neutral SMPL-X and assumes 30 fps. Thus provide CPU, detached tensors with the expected axes/units; a camera-space dictionary is not a substitute. Other GVHMR result fields are not consumed by this loader. Output keys/quaternion convention match the SMPL-X script. [Loader][GMR-utils] [GVHMR script][GMR-gvhmr]

**Sources / Date:** pinned author source above; 2026-10-28. No retargeting execution claimed.

## D-04 MuJoCo Menagerie Unitree G1

**Decision / Options / Choice / Why.** Use Menagerie's `unitree_g1/g1.xml` without dexterous hands as the canonical rendering/physics asset. Map GMR coordinates by joint name and check against the canonical ranges; do not assume the two XMLs are identical. Menagerie's model is BSD-3-Clause, allowing public redistribution with copyright/conditions/disclaimer and no endorsement. [Model licence][MG-license] [Author derivation][MG-readme]

**Download and size.** Obtain the `unitree_g1` directory from official Menagerie at inspected revision `f3b6013629c7b94f8ee93fd977b161fa8350f9bc`. Its entire subtree is 64 files / 39,655,929 bytes by Git tree blob sizes, including extra variants/images; actual browser packaging can be smaller and is unmeasured. Include the referenced `assets/` meshes and licence. [Tree API][MG-tree] [MJCF mesh references][MG-xml]

**Exact source comparison.** Both files have the same 29 hinge names/traversal order and the same joint-level `actuatorfrcrange` values. Ranges differ. The table is derived directly from the two pinned XMLs; values are radians and joint generalized force limits in N m, not hardware certification or continuous-duty ratings. [Menagerie XML][MG-xml] [GMR XML][GMR-xml] [Audit](research/stack-source-audit-2026-10-28.json)

| Joint (MJCF traversal order) | Menagerie range (rad) | GMR range (rad) | Both joint torque limits (N m) |
| --- | --- | --- | --- |
| `left_hip_pitch_joint` | `-2.5307 2.8798` | `-1.57 1.57` | `-88 88` |
| `left_hip_roll_joint` | `-0.5236 2.9671` | `-0.5236 1.57` | `-139 139` |
| `left_hip_yaw_joint` | `-2.7576 2.7576` | `-1.57 1.57` | `-88 88` |
| `left_knee_joint` | `-0.087267 2.8798` | `-0.087267 2.8798` | `-139 139` |
| `left_ankle_pitch_joint` | `-0.87267 0.5236` | `-0.87267 0.5236` | `-50 50` |
| `left_ankle_roll_joint` | `-0.2618 0.2618` | `-0.2618 0.2618` | `-50 50` |
| `right_hip_pitch_joint` | `-2.5307 2.8798` | `-1.57 1.57` | `-88 88` |
| `right_hip_roll_joint` | `-2.9671 0.5236` | `-1.57 0.5236` | `-139 139` |
| `right_hip_yaw_joint` | `-2.7576 2.7576` | `-1.57 1.57` | `-88 88` |
| `right_knee_joint` | `-0.087267 2.8798` | `-0.087267 2.8798` | `-139 139` |
| `right_ankle_pitch_joint` | `-0.87267 0.5236` | `-0.87267 0.5236` | `-50 50` |
| `right_ankle_roll_joint` | `-0.2618 0.2618` | `-0.2618 0.2618` | `-50 50` |
| `waist_yaw_joint` | `-2.618 2.618` | `-1.57 1.57` | `-88 88` |
| `waist_roll_joint` | `-0.52 0.52` | `-0.52 0.52` | `-50 50` |
| `waist_pitch_joint` | `-0.52 0.52` | `-0.52 0.52` | `-50 50` |
| `left_shoulder_pitch_joint` | `-3.0892 2.6704` | `-3.0892 1.149` | `-25 25` |
| `left_shoulder_roll_joint` | `-1.5882 2.2515` | `-0.6 2.2515` | `-25 25` |
| `left_shoulder_yaw_joint` | `-2.618 2.618` | `-1.4 2.0` | `-25 25` |
| `left_elbow_joint` | `-1.0472 2.0944` | `-1.0472 1.7` | `-25 25` |
| `left_wrist_roll_joint` | `-1.97222 1.97222` | `-1.97222 1.97222` | `-25 25` |
| `left_wrist_pitch_joint` | `-1.61443 1.61443` | `-1.61443 1.61443` | `-5 5` |
| `left_wrist_yaw_joint` | `-1.61443 1.61443` | `-1.61443 1.61443` | `-5 5` |
| `right_shoulder_pitch_joint` | `-3.0892 2.6704` | `-3.0892 1.149` | `-25 25` |
| `right_shoulder_roll_joint` | `-2.2515 1.5882` | `-2.2515 0.6` | `-25 25` |
| `right_shoulder_yaw_joint` | `-2.618 2.618` | `-2.0 1.4` | `-25 25` |
| `right_elbow_joint` | `-1.0472 2.0944` | `-1.0472 1.7` | `-25 25` |
| `right_wrist_roll_joint` | `-1.97222 1.97222` | `-1.97222 1.97222` | `-25 25` |
| `right_wrist_pitch_joint` | `-1.61443 1.61443` | `-1.61443 1.61443` | `-5 5` |
| `right_wrist_yaw_joint` | `-1.61443 1.61443` | `-1.61443 1.61443` | `-5 5` |

**Controller and feasibility gotchas.** Menagerie uses position actuators with default `kp=500`, `dampratio=1`, `inheritrange=1`; its README says these need tuning. GMR uses motors with inherited `ctrlrange="-1 1"` and separate joint torque limits. Therefore identical qpos does not imply identical closed-loop dynamics, and the GMR motor control range is not the same quantity as the tabulated torque capacity. Choose Menagerie dynamics for feasibility evaluation and record every actuator/contact change. Validate joint names, axes, body transforms and root conventions before claiming equivalent kinematics; this audit compared names/ranges/torque metadata, not full physical equivalence. [Menagerie defaults][MG-xml] [Derivation warning][MG-readme] [GMR defaults][GMR-xml]

**Sources / Date:** above; 2026-10-28.

## D-05 Official MuJoCo JavaScript/WASM

**Decision / Options / Choice / Why.** Choose the official Google DeepMind npm package, pinned initially to `@mujoco/mujoco@3.9.0`, not an unpinned fork. Apache-2.0 allows public demo redistribution with applicable notices. Version 3.9.0 is a verified baseline, not claimed to be latest. [Publisher package metadata][MJ-npm] [Canonical bindings README][MJ-readme]

**Verified capabilities from the published tarball's TypeScript declarations:** `MjVFS.addBuffer(name, bytes)`, `MjModel.from_xml_string(xml, vfs)` and `from_xml_path(path, vfs)` support supplying an MJCF and its assets; `mj_forward`, `mj_kinematics`, `mj_fwdPosition`, `mj_comPos`, `mj_jac` and `mj_jacSite` are exposed. Thus the API can load MJCF with meshes and calculate site Jacobians in a browser. Fetch each mesh/texture and register its matching path in the VFS before compilation; HTTP URLs in the XML are not automatically a network filesystem. This is source/package verification, not a successful G1 browser run. [Published package][MJ-tgz] [Typed declarations][MJ-types]

For IK, update qpos, run `mj_forward` or the required position pipeline before Jacobians. Allocate suitable output buffers and release Embind handles; 3.9.0 declarations use `DoubleBuffer`, `FromArray` and `GetView`, so follow the pinned declarations rather than assuming examples from main match every release. [Typed API][MJ-types] [Memory guidance][MJ-readme]

| Published 3.9.0 artifact | Exact bytes |
| --- | ---: |
| npm `.tgz` download (all included builds) | 5,047,889 |
| Entire package unpacked | 19,723,229 |
| Default single-thread `mujoco.wasm` | 9,139,270 |
| Default single-thread `mujoco.js` | 287,485 |
| Multi-thread `mt/mujoco.wasm` | 9,264,885 |

Source: [npm registry][MJ-npm], inspected [official tarball][MJ-tgz]; recorded in [audit](research/stack-source-audit-2026-10-28.json). These are package bytes, not the total app download, mesh budget, runtime RAM, or browser HTTP-compressed size.

**Performance.** No verified end-to-end G1 + detector + student + IK laptop fps result was found or measured. Treat the project fps target as unproven. The default build is single-threaded; `/mt` requires SharedArrayBuffer and cross-origin isolation (COOP/COEP). Choose single-thread initially so demo availability does not depend on multi-thread headers; benchmark later under the feature's protocol. [Official threading and benchmark guidance][MJ-readme]

**Sources / Date:** above; 2026-10-28.

## D-06 ONNX Runtime Web

**Decision / Options / Choice / Why.** Choose `onnxruntime-web` with WebGPU preferred and an explicit WASM-only recovery path. The inspected npm release is 1.31.1 (MIT); retain notices. Import the WebGPU entry, request `executionProviders: ['webgpu', 'wasm']`, and recreate a WASM session if device initialization fails. Provider ordering is not a promise that every device or exported graph will work. [WebGPU guide][ORT-gpu] [MIT][ORT-license] [Registry][ORT-npm]

| Topic | Verified support / limitation | Project choice |
| --- | --- | --- |
| FP16 | Browser WebGPU implementation requests `shader-f16` when available and has FP16 paths. Device/browser support is required. | FP16 export only after feature detection and parity tests; retain FP32. [Backend source][ORT-backend] |
| INT8 / QDQ | WASM inherits ORT CPU quantization. Browser WebGPU coverage is operator/build dependent: the inspected JS resolver exposes `DequantizeLinear`, not a blanket complete INT8/QDQ GPU path. The newer native WebGPU provider must not be confused with that resolver. | Evaluate INT8 QDQ on WASM; never promise GPU INT8 acceleration or speedup without profiling the pinned build and checking node placement. [Web README][ORT-readme] [Versioned JS resolver][ORT-resolver] [Build selection][ORT-import] |
| Unsupported GPU operators | WebGPU support is a subset of ONNX; CPU execution can be required. | Validate each exported graph, including preprocessing/postprocessing; retain a known-good FP32 WASM model. [Operator table][ORT-ops] [Web guide][ORT-web] |
| WASM threads | Require browser thread support and `crossOriginIsolated`; `numThreads=1` disables threads. | Use COOP `same-origin` plus COEP `require-corp` (with suitable asset CORS/CORP), or single-thread fallback. Verify `crossOriginIsolated` at runtime. [ORT flags][ORT-flags] [Isolation guide][Isolation] |

**Download/size.** Install the pinned npm release and self-host its matching JS/WASM assets. Registry `dist.unpackedSize` is 150,622,959 bytes for the whole 1.31.1 package; this includes multiple builds and is not the delivered browser bundle size. Exact chosen app transfer size and latency are unmeasured. JS and WASM versions must match. [Registry][ORT-npm] [Deployment guide][ORT-deploy]

**Gotchas.** Keep a worker responsive path; ORT's proxy worker is not generally interchangeable with the WebGPU worker arrangement. Serve LIVE assets locally/same-origin where possible and test network activity against the no-camera-upload rule. Quantization validation must use training/validation data, never test-set decisions. These are project requirements informed by [ORT flags][ORT-flags] and the [constitution](../.specify/memory/constitution.md).

**Sources / Date:** above; 2026-10-28. No exported model or browser performance verified.

## D-07 Real-world evaluation: EMDB, 3DPW and RICH

**Decision / Options / Choice / Why.** Choose **3DPW as the one primary real-world benchmark**, offline and evaluation-only. Its official split protocol and public download sizes are verifiable without a pending institutional application. EMDB remains a useful future global-motion benchmark; RICH is a future contact/scene benchmark. This is a scope choice, not an accuracy comparison. [3DPW protocol][PW-eval] [EMDB evaluation][EMDB-readme] [RICH paper][RICH-paper]

| Dataset | Licence and public portfolio boundary | Ground truth / standard protocol | Download method and size |
| --- | --- | --- | --- |
| EMDB | Non-commercial academic use; no redistribution, including converted model/data parameters; prohibits commercial promotion/direct or indirect commercial gain. Keep raw media offline; any portfolio use beyond academic reporting needs clarification. [Terms][EMDB-terms] | Paper: 81 sequences, 10 participants, 58 minutes; electromagnetic sensors plus video produce SMPL pose/shape, global body and camera trajectories. `emdb1`/`emdb2` flags and `good_frames_mask` identify evaluation subsets/valid frames. Official EMDB1 evaluator reports pelvis/Procrustes-aligned MPJPE, MPJAE, MVE and jitter; EMDB2 addresses global trajectories. [Paper][EMDB-paper] [Schema][EMDB-schema] [Evaluator README][EMDB-readme] | Institutional-email application; approved download instructions arrive by email. **Exact archive list and byte size unverified** without approval. Images, per-sequence pickle and visualization video are documented; visualization MP4 contains an overlay, so it is not interchangeable with raw RGB input. [Access][EMDB-terms] [Schema][EMDB-schema] |
| 3DPW | Non-commercial scientific research only; no reproducing/modifying/making data available to third parties without written permission. Use for offline evaluation and cited numerical reporting, not bundled demo clips or meshes. [Licence][PW-license] | 60 moving-camera sequences; video+IMU-derived 3D SMPL pose/shape, 2D poses and per-frame camera poses. Official `Train-Test-mode` uses train/validation/test directories; `All-Test-mode` evaluates all directories. State the chosen mode explicitly. [Dataset][PW-home] [Protocols][PW-eval] | Official licence page links `imageFiles.zip` **4,610,672,154 bytes**, `sequenceFiles.zip` **277,814,416 bytes**, `readme_and_demo.zip` **4,539 bytes**; verified by HTTP Content-Length, not download/checksum validation. No dataset account is requested by this public page; SMPL access is separate. [Images][PW-images] [Sequences][PW-sequences] [README archive][PW-demo] [Licence/download page][PW-license] |
| RICH | Non-commercial research/education/artistic purposes; no redistribution or third-party availability without written permission. Do not ship test media or body files in the demo. [Licence][RICH-license] | Markerless 3D SMPL-X bodies, calibrated multiview imagery, scene scans and dense contact labels. Paper reports 134 videos, 85K posed meshes/contact annotations and 540K images; its original contact protocol uses precision/recall/F1. Toolkit preserves official train/val/test and camera-to-world transforms. No single universal mesh-evaluation protocol was verified; state joint set/alignment if using it later. [Paper][RICH-paper] [Toolkit][RICH-toolkit] | Register at the official portal, then select authorized image/body/contact/calibration archives. **Exact archive list and byte size unverified** behind registration. [Registration][RICH-register] [Toolkit download layout][RICH-toolkit] |

**Primary protocol choice.** Report 3DPW **official test directory under Train-Test-mode**, with no 3DPW training or model selection in this project; all choices use BEDLAM validation. Freeze the joint mapping, alignment, valid-frame mask and metrics before inspecting test results. Report pelvis-aligned MPJPE and PA-MPJPE with the precise mapping/protocol; add mesh metrics only after a valid licensed SMPL-X-to-SMPL comparison path exists. The official page recommends joint and mesh errors, but does not make every paper's regressor/alignment convention identical. Do not label this `All-Test-mode` or compare to that mode without qualification. [Official protocols/metrics][PW-eval] [Constitution](../.specify/memory/constitution.md)

**Gotchas.** SMPL ground truth is not natively SMPL-X topology. An arbitrary joint subset or retargeted robot error is not a substitute for the published human-pose metric. Obtain the SMPL model through its own registration, and verify any conversion/regressor file's licence before use. Do not tune on a sequence because its frames are in a public download. [3DPW requirements][PW-eval] [SMPL model access][SMPL-home] [SMPL-X model family/download instructions][SX-readme]

**Sources / Date:** above; 2026-10-28.

## D-08 Browser person detector

**Decision / Options / Choice / Why.** Choose **RTMDet-tiny COCO**, export to ordinary ONNX and use only the person category, through the same ORT runtime as the student. RTMPose is a pose estimator, not a substitute for the person detector; using it as the main pose network would also weaken the intended student-training contribution. This is a project choice, not a browser speed result. [RTMDet model zoo/export][RTM-det] [RTMPose project][RTM-pose]

| Option | Licence / public demo | Download / size | Gotchas and disposition |
| --- | --- | --- | --- |
| Ultralytics YOLO11n / YOLO26n-pose | AGPL-3.0 or Enterprise. **A public repo alone is insufficient**: applicable AGPL licensing, corresponding source, notices and network-interaction obligations must be met. Ultralytics' own guidance asks the larger derivative application to be AGPL, including frontend/backend and relevant training/config files. AGPL does not itself forbid commercial use; Enterprise is the alternative for incompatible closed-source use. [Vendor licensing guidance][YOLO-license] [AGPL text][AGPL] | Official package loads `.pt` then exports ONNX. YOLO26 supports detection and pose variants. Exact chosen ONNX bytes are **unverified** because no export was performed. [YOLO26][YOLO26] [Export][YOLO-export] | `YOLO11n` detector and `YOLO26n-pose` have different outputs/tasks. ONNX export does not erase licence obligations. Reject for this stack to avoid copyleft integration complexity; do not relicense restricted training data to satisfy it. [Vendor guidance][YOLO-license] |
| MediaPipe Object Detector + EfficientDet-Lite0 | SDK is Apache-2.0. The exact EfficientDet-Lite object-detector weight grant was **not established** by the SDK licence alone. A maintainer explicitly confirmed Apache-2.0 for a particular Pose Landmarker Full task, a different artifact. [SDK licence][MP-license] [Maintainer clarification][MP-issue] | Official direct TFLite downloads in int8/float16/float32; the guide specifies Lite0 input 320x320. Exact selected bytes not verified. [Model links][MP-models] | Convenient Web API, but adds a second inference stack; weight-specific clarification is needed before choosing the detector artifact. `detectForVideo` blocks its calling thread; use a worker. Do not infer that pose-task confirmation covers EfficientDet weights. [Web guide][MP-web] [Issue][MP-issue] |
| RTMDet / RTMPose | MMDetection and MMPose are Apache-2.0; official model-zoo checkpoints are distributed from those projects. The recommendation relies on that project-wide release licence; a separate checkpoint-specific grant was not located. Record that inference and confirm it with the publisher if needed before redistribution. Keep licence/NOTICE and document upstream checkpoint identity; no AGPL requirement. Dataset and any third-party notices still apply. [MMDetection licence][RTM-license] [MMPose licence][RTMP-license] [Official checkpoint table][RTM-det] | Selected RTMDet-tiny `.pth` is **57,532,893 bytes** by official-host HEAD; trained/exported ONNX size is not measured. Download from the official model table and export with supported MMDeploy/ONNX tooling, verifying absence of unsupported custom ops. RTMPose has separate author deployment examples; no RTMPose weight is selected. [Checkpoint][RTM-weight] [RTMDet export][RTM-det] [RTMPose deployment][RTMP-deploy] | Browser compatibility and target fps still need profiling. Select standard ONNX outputs and perform person filtering/NMS with tested postprocessing rather than assuming every desktop deployment plugin works in WebGPU. This is a proposed integration requirement. [Export docs][RTM-det] [ORT operator coverage][ORT-ops] |

**Sources / Date:** above; 2026-10-28.

## D-09 Teacher and student backbones

**Decision / Options / Choice / Why.** Choose **DINOv2 ViT-B/14** for the teacher image encoder, with our own temporal module and pose/shape/camera heads; choose **`mobilenetv4_conv_small.e2400_r224_in1k` via timm** for the student encoder. Their named weight releases explicitly use Apache-2.0. This keeps the ML contribution in the trained teacher/student/distillation system, while avoiding a research-only ready-made HMR checkpoint. The choice is based on licence clarity and model structure, not measured superiority. [DINOv2 release][DINO2] [Base weight card][DINO2-base] [MobileNetV4 card][MB4]

| Candidate | Licence and public non-commercial demo permission | Official download / size | Gotchas / choice |
| --- | --- | --- | --- |
| DINOv2 ViT-S/14 and ViT-B/14 | Ordinary DINOv2 code **and model weights**: Apache-2.0; usable in a public demo with notices. Do not generalize this to newer CELL-DINO/X-ray entries in the same repository, which carry other licences. [README licence sections][DINO2] [S card][DINO2-small] [B card][DINO2-base] | `torch.hub` links in official README, or publisher HF repos `facebook/dinov2-small` / `facebook/dinov2-base`. HF `model.safetensors`: S **88,249,960 bytes**, B **346,345,912 bytes**, excluding config/preprocessor files. [S manifest][DINO2-small-files] [B manifest][DINO2-base-files] | Choose B; S is the fallback encoder size. These are feature backbones, not SMPL-X/video estimators without heads/temporal training. [Model cards][DINO2-base] |
| DINOv3 | Custom DINOv3 licence, **not Apache/MIT** and not simply research-only: grants use/derivatives/distribution, but requires redistribution under the agreement, publication acknowledgement, legal/privacy/trade-control compliance, bans reverse engineering/discovering underlying components and specified prohibited end uses, and has termination/amendment provisions. [Exact licence][DINO3-license] | Meta access request/acceptance or gated publisher HF checkpoints; size depends on variant. Exact selected checkpoint size **unverified** because none is selected. [Official model/download instructions][DINO3] | Non-commercial status does not remove these conditions. Reject to keep a simpler permissive-licence chain. No blanket ban on ordinary commercial use or invented user-count threshold is asserted. [Licence][DINO3-license] |
| HMR2.0 / 4D-Humans | Repository code is MIT. That does not independently prove every externally downloaded checkpoint/body-model file is MIT. A separate weight-specific grant for all `hmr2_data.tar.gz` contents was **not verified**; SMPL explicitly requires separate registration. [MIT][HMR-license] [README][HMR-readme] | Author loader downloads `hmr2_data.tar.gz`; HTTP HEAD returned **2,710,589,440 bytes** for the bundle, not the backbone alone. [Loader][HMR-loader] [Bundle][HMR-bundle] | Useful reference/baseline only after checking all bundle terms. Do not choose it as a supposedly clean generic pretrained backbone. [Setup][HMR-readme] |
| CameraHMR / BEDLAM2 checkpoint | Official licence explicitly includes checkpoints/scripts and limits use to non-commercial research/education/art, with no distribution absent permission. Not a clean permissive release. [Licence][CAM-license] | Register at CameraHMR, BEDLAM2 and SMPL-X for the documented SMPL-X demo. Script names `bedlam_v1_v2.ckpt`, `cam_model_cleaned.ckpt`, `model_final_f05665.pkl`, `smpl_mean_params.npz`; exact sizes **unverified** behind gates. [Author downloader][CAM-download] [SMPL-X instructions][CAM-b2] | Reject for public bundling; source availability is not an unrestricted licence. Do not assume its SMPL output and SMPL-X variant use the same labels/model. [README][CAM-readme] |
| ViTPose weights | Repository is Apache-2.0; pretrained MAE and task checkpoint provenance must be checked separately. A single explicit grant covering every linked OneDrive checkpoint and its underlying pretraining assets was **not verified** here. [Licence][VIT-license] [Weight links][VIT-readme] | Official README links small MAE and task weights on OneDrive; B/L/H MAE to the MAE project. Exact artifact bytes **unverified**. [README][VIT-readme] | 2D keypoint model, not a ready temporal SMPL-X teacher. Do not choose an unspecified ViTPose weight merely because the repository is Apache. [README][VIT-readme] |
| MobileNetV4 Conv Small | Selected timm weight card: Apache-2.0. [Card][MB4] | `timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=True)` or publisher HF; safetensors **15,223,016 bytes**. [Card][MB4] [Manifest][MB4-files] | **Chosen student**; replace classification head and train/distil. Final ONNX bytes and performance depend on our heads/export/precision and remain unmeasured. [Card feature-extraction examples][MB4] |
| EfficientViT B0 | Named MIT Han Lab family and selected timm card are Apache-2.0; this entry is specifically `efficientvit_b0.r224_in1k`, not every similarly named EfficientViT release. [Author licence][EVIT-license] [timm card][EVIT] | timm/HF; safetensors **13,684,888 bytes**. [Manifest][EVIT-files] | Permissive alternative; no browser latency advantage established. [Card][EVIT] |
| RepViT M1.0 | Author repository and named timm card are Apache-2.0. [Author licence][REP-license] [timm card][REP] | `repvit_m1_0.dist_450e_in1k` through timm/HF; safetensors **29,519,376 bytes**. [Manifest][REP-files] | Permissive alternative; structural reparameterization requires an export/reparameterization check, not an assumption that the training graph is the best inference graph. [Author instructions][REP-readme] |

**Licence boundary.** Clean backbone weights do not make a BEDLAM-trained teacher or distilled student automatically safe to publish. Keep the resulting checkpoints private until D-01/D-02 permissions are established. Preserve the exact backbone model cards, versions and notices with the release record. [BEDLAM terms][B2-license] [SMPL-X terms][SX-license]

**Sources / Date:** above; 2026-10-28. Checkpoint sizes are publisher file metadata, not estimates from parameter counts.

## D-10 Hugging Face hosting and ZeroGPU

**Decision / Options / Choice / Why.** Choose a free Static HTML Space for the LIVE frontend and a separate public Gradio ZeroGPU Space for RECORD **only after licence clearance**. Keep a local demo replay available when camera/GPU/server access fails. PRO is optional unless needed for eligibility/capacity; do not purchase it automatically. [Static hosting][HF-static] [Free static confirmation][HF-gpu] [ZeroGPU][HF-zero]

| Current ZeroGPU rule | Verified value |
| --- | --- |
| Free hosting | Up to **2** ZeroGPU Spaces for personal accounts in good standing: verified email, account older than **30 days** |
| PRO hosting | Up to **10** ZeroGPU Spaces |
| Default GPU call duration | **60 seconds**, configurable; not a promise of always fitting a video request |
| Visitor daily quota | Unauthenticated **2 min**; free account **5 min**; PRO **40 min**, extensible; per visitor account, not one owner's allowance for everyone |
| SDK/runtime | Gradio only; Gradio **4+**; supported PyTorch list starts at **2.8.0**; Python **3.10.13 / 3.12.12** |
| Compilation | `torch.compile` unsupported; ahead-of-time compilation is available with supported PyTorch |
| Default hardware | `large`: half RTX Pro 6000 Blackwell, **48 GB**; `xlarge`: **96 GB**, twice quota cost |

Source for the table: [current official ZeroGPU documentation][HF-zero]. These are dated published service limits, not observed account entitlements. Request duration affects quota admission/queueing; choose it from measured workload duration rather than leaving a needlessly large default. [Quota handling][HF-quota]

**Price and static hosting.** PRO is currently **US$9/month**; Static Spaces are free for everyone. Static supports HTML/JS/CSS and optional build commands, not server GPU Python. A plain static frontend still needs tested camera permissions and appropriate headers if multithreaded WASM is enabled; do not assume an embedded Space guarantees isolation. [Price][HF-price] [Free/static distinction][HF-gpu] [Static SDK][HF-static] [ORT threads][ORT-flags]

**Research-only trained weights.** HF supplies hosting, not a blanket rights grant for models trained on third-party research data. Its terms make the user responsible for rights to uploaded content. Public hosting is acceptable only when the upstream licences/permissions cover that exact use; no official source was found universally approving weights trained on research-only data. This stack's BEDLAM/SMPL-X public-use question remains unresolved. [HF terms][HF-terms] [BEDLAM licence][B2-license] [SMPL-X licence][SX-license]

**Download/size and operational gotchas.** Deploy code through the Space repo; model bytes are those selected in D-09 plus our as-yet-untrained heads/checkpoint. There is no verified final Space download size. Fine-grained private-weight read tokens belong in server-side Secrets, never frontend variables. Upload deletion must include success, exception, cancellation and Gradio temporary/cache paths; this research does not certify a running cleanup implementation. [Space configuration/secrets][HF-overview] [Token permissions][HF-tokens] [Constitution](../.specify/memory/constitution.md)

**Sources / Date:** above; 2026-10-28.

## D-00 Overall stack

**Decision.** Choose this research stack, with public release conditional on the bold permission gates at the top. These are architecture choices from the cited entries, not feature implementation or benchmark results.

| Layer | Chosen stack | Alternatives / reason |
| --- | --- | --- |
| Training data | BEDLAM2 selected MP4 archives, matching camera/scene metadata, registered motion NPZ and locked-head model | Exclude PNG/depth; do not assume CSV/JSON alone supplies SMPL-X trajectories. [D-01](#d-01-bedlam-20) [Author preprocessing][B2-process] |
| Teacher | DINOv2 ViT-B/14 + our temporal and SMPL-X/shape/camera heads | Prefer explicit Apache backbone over ready-made research-restricted CameraHMR weights. [DINOv2][DINO2] [CameraHMR terms][CAM-license] |
| Student | timm MobileNetV4 Conv Small, trained by distillation | EfficientViT and RepViT remain alternatives; no speed ranking claimed. [Chosen model card][MB4] |
| Detector | RTMDet-tiny, official checkpoint, standard ONNX, person filter | Avoid AGPL integration obligations and a second runtime. [Model zoo][RTM-det] [Licensing comparison][YOLO-license] |
| Browser inference | ORT Web 1.31.1; WebGPU, FP32 WASM recovery; FP16/INT8 only after compatibility and parity checks | No automatic claim of QDQ GPU acceleration. [ORT guide][ORT-gpu] [Quantization support][ORT-readme] |
| Browser robot | Official MuJoCo WASM 3.9.0 baseline + Menagerie G1 + three.js | Menagerie/GMR models require explicit mapping. three.js is MIT; keep its notice. [MuJoCo package][MJ-npm] [Menagerie][MG-readme] [three.js licence][THREE-license] |
| Record retargeting | GMR pinned source; explicit SMPL-X adapter and Menagerie feasibility checks | GVHMR file format documented as an alternative, not selected training weights. [GMR loader][GMR-utils] [Robot comparison][MG-xml] |
| Primary real test | 3DPW official test under disclosed Train-Test-mode; offline only | EMDB/RICH deferred; do not use test data for model selection. [Official protocol][PW-eval] |
| Hosting | Free static LIVE + Gradio ZeroGPU RECORD; optional PRO; separate supported Space Python | Public teacher/student release and private body-model hosting await rights-holder clearance. [ZeroGPU][HF-zero] [SMPL-X terms][SX-license] |

**Required permission decision:** ask BEDLAM/SMPL-X rights holders to explicitly cover (a) internship-oriented public non-commercial portfolio use, (b) server inference on visitor video, (c) distribution of trained teacher and distilled browser ONNX weights, (d) licensed body files stored in a private HF repo and loaded by a public Space, and (e) permitted output/replay sharing. If refused, retain private research and use only independently cleared models/assets for the public fallback; do not disguise that fallback as our trained student. [Unresolved scope of BEDLAM grant][B2-license] [SMPL-X grant][SX-license]

**Known unverified items:** gated BEDLAM motion/metadata/body-model bytes and canonical body/motion IDs; public deployment/derived-weight permission; final browser throughput and bundle sizes; GMR beta/body-model runtime compatibility; complete G1 physical equivalence; weight-specific grants for unselected HMR/ViTPose/MediaPipe detector artifacts. These are explicit research limitations, not invented facts. Sources and access limits are recorded in D-01 through D-10.

**Sources / Date:** all linked entries; 2026-10-28. No spec folder, feature plan, feature code or cluster job created.

<!-- Primary-source URL definitions. -->
[B2-license]: https://bedlam2.is.tue.mpg.de/license.html
[B2-home]: https://bedlam2.is.tue.mpg.de/
[B2-hf]: https://huggingface.co/datasets/Intelligent-Systems/BEDLAM2
[B2-tree]: https://huggingface.co/datasets/Intelligent-Systems/BEDLAM2/tree/15e18597eb570e1d7166a0b9ef1009d35b904093
[B2-api]: https://huggingface.co/api/datasets/Intelligent-Systems/BEDLAM2/tree/15e18597eb570e1d7166a0b9ef1009d35b904093?recursive=true&limit=1000
[B2-download]: https://bedlam2.is.tue.mpg.de/download.php
[B2-process]: https://github.com/pixelite1201/BEDLAM/blob/master/data_processing/ReadMe.md
[B2-parser]: https://github.com/pixelite1201/BEDLAM/blob/master/data_processing/df_full_body_smplx_bedlam2.py
[B2-download-script]: https://github.com/pixelite1201/BEDLAM/blob/master/data_processing/download_utils_bedlam2.sh
[B2-scenes]: https://github.com/pixelite1201/BEDLAM/blob/master/data_processing/bedlam2_scene_names.csv
[B2-faq]: https://bedlam2.is.tue.mpg.de/faq.html
[SX-license]: https://smpl-x.is.tue.mpg.de/modellicense.html
[SX-download]: https://smpl-x.is.tue.mpg.de/download.php
[SX-readme]: https://github.com/vchoutas/smplx#downloading-the-model
[SMPL-home]: https://smpl.is.tue.mpg.de/
[GMR-readme]: https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/README.md
[GMR-utils]: https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/general_motion_retargeting/utils/smpl.py
[GMR-script]: https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/scripts/smplx_to_robot.py
[GMR-gvhmr]: https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/scripts/gvhmr_to_robot.py
[GMR-xml]: https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/assets/unitree_g1/g1_mocap_29dof.xml
[GMR-params]: https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/general_motion_retargeting/params.py
[GMR-license]: https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/LICENSE
[GMR-tree]: https://api.github.com/repos/YanjieZe/GMR/git/trees/25f56db79ac411258b377b6d4ccba0c8ff3372ac?recursive=1
[MG-xml]: https://github.com/google-deepmind/mujoco_menagerie/blob/f3b6013629c7b94f8ee93fd977b161fa8350f9bc/unitree_g1/g1.xml
[MG-readme]: https://github.com/google-deepmind/mujoco_menagerie/blob/f3b6013629c7b94f8ee93fd977b161fa8350f9bc/unitree_g1/README.md
[MG-license]: https://github.com/google-deepmind/mujoco_menagerie/blob/f3b6013629c7b94f8ee93fd977b161fa8350f9bc/unitree_g1/LICENSE
[MG-tree]: https://api.github.com/repos/google-deepmind/mujoco_menagerie/git/trees/f3b6013629c7b94f8ee93fd977b161fa8350f9bc?recursive=1
[MJ-npm]: https://registry.npmjs.org/@mujoco/mujoco/3.9.0
[MJ-tgz]: https://registry.npmjs.org/@mujoco/mujoco/-/mujoco-3.9.0.tgz
[MJ-types]: https://unpkg.com/@mujoco/mujoco@3.9.0/mujoco.d.ts
[MJ-readme]: https://github.com/google-deepmind/mujoco/blob/main/wasm/README.md
[ORT-license]: https://github.com/microsoft/onnxruntime/blob/v1.31.1/LICENSE
[ORT-npm]: https://registry.npmjs.org/onnxruntime-web/1.31.1
[ORT-gpu]: https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html
[ORT-web]: https://onnxruntime.ai/docs/get-started/with-javascript/web.html
[ORT-backend]: https://github.com/microsoft/onnxruntime/blob/main/js/web/lib/wasm/jsep/backend-webgpu.ts
[ORT-resolver]: https://github.com/microsoft/onnxruntime/blob/v1.31.1/js/web/lib/wasm/jsep/webgpu/op-resolve-rules.ts
[ORT-import]: https://github.com/microsoft/onnxruntime/blob/v1.31.1/js/web/lib/wasm/wasm-utils-import.ts
[ORT-readme]: https://github.com/microsoft/onnxruntime/blob/main/js/web/README.md
[ORT-ops]: https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md
[ORT-flags]: https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html
[ORT-deploy]: https://onnxruntime.ai/docs/tutorials/web/deploy.html
[Isolation]: https://web.dev/articles/cross-origin-isolation-guide
[EMDB-terms]: https://emdb.ait.ethz.ch/
[EMDB-paper]: https://arxiv.org/abs/2308.16894
[EMDB-schema]: https://github.com/eth-ait/emdb/blob/main/dataset.md
[EMDB-readme]: https://github.com/eth-ait/emdb/blob/main/README.md
[PW-home]: https://virtualhumans.mpi-inf.mpg.de/3DPW/
[PW-license]: https://virtualhumans.mpi-inf.mpg.de/3DPW/license.html
[PW-eval]: https://virtualhumans.mpi-inf.mpg.de/3DPW/evaluation.html
[PW-images]: https://virtualhumans.mpi-inf.mpg.de/3DPW/imageFiles.zip
[PW-sequences]: https://virtualhumans.mpi-inf.mpg.de/3DPW/sequenceFiles.zip
[PW-demo]: https://virtualhumans.mpi-inf.mpg.de/3DPW/readme_and_demo.zip
[RICH-license]: https://rich.is.tue.mpg.de/license.html
[RICH-register]: https://rich.is.tue.mpg.de/register.php
[RICH-paper]: https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Capturing_and_Inferring_Dense_Full-Body_Human-Scene_Contact_CVPR_2022_paper.pdf
[RICH-toolkit]: https://github.com/paulchhuang/rich_toolkit
[YOLO-license]: https://docs.ultralytics.com/help/contributing/#open-sourcing-your-yolo-project-under-agpl-30
[AGPL]: https://www.gnu.org/licenses/agpl-3.0.html
[YOLO26]: https://docs.ultralytics.com/models/yolo26/
[YOLO-export]: https://docs.ultralytics.com/modes/export/
[MP-license]: https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE
[MP-issue]: https://github.com/google-ai-edge/mediapipe/issues/6306
[MP-models]: https://developers.google.com/edge/mediapipe/solutions/vision/object_detector
[MP-web]: https://developers.google.com/edge/mediapipe/solutions/vision/object_detector/web_js
[RTM-det]: https://github.com/open-mmlab/mmdetection/blob/main/configs/rtmdet/README.md
[RTM-pose]: https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose/README.md
[RTM-license]: https://github.com/open-mmlab/mmdetection/blob/main/LICENSE
[RTMP-license]: https://github.com/open-mmlab/mmpose/blob/main/LICENSE
[RTMP-deploy]: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose/examples/onnxruntime
[RTM-weight]: https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth
[DINO2]: https://github.com/facebookresearch/dinov2#license
[DINO2-base]: https://huggingface.co/facebook/dinov2-base
[DINO2-small]: https://huggingface.co/facebook/dinov2-small
[DINO2-base-files]: https://huggingface.co/api/models/facebook/dinov2-base/tree/main
[DINO2-small-files]: https://huggingface.co/api/models/facebook/dinov2-small/tree/main
[DINO3]: https://github.com/facebookresearch/dinov3
[DINO3-license]: https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md
[HMR-license]: https://github.com/shubham-goel/4D-Humans/blob/main/LICENSE.md
[HMR-readme]: https://github.com/shubham-goel/4D-Humans
[HMR-loader]: https://github.com/shubham-goel/4D-Humans/blob/main/hmr2/models/__init__.py
[HMR-bundle]: https://www.cs.utexas.edu/~pavlakos/4dhumans/hmr2_data.tar.gz
[CAM-license]: https://camerahmr.is.tue.mpg.de/license.html
[CAM-readme]: https://github.com/pixelite1201/CameraHMR/blob/master/ReadMe.md
[CAM-download]: https://github.com/pixelite1201/CameraHMR/blob/master/scripts/fetch_demo_data_bedlam2.sh
[CAM-b2]: https://github.com/pixelite1201/CameraHMR/blob/master/docs/bedlam2.md
[VIT-license]: https://github.com/ViTAE-Transformer/ViTPose/blob/main/LICENSE
[VIT-readme]: https://github.com/ViTAE-Transformer/ViTPose
[MB4]: https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k
[MB4-files]: https://huggingface.co/api/models/timm/mobilenetv4_conv_small.e2400_r224_in1k/tree/main
[EVIT]: https://huggingface.co/timm/efficientvit_b0.r224_in1k
[EVIT-license]: https://github.com/mit-han-lab/efficientvit/blob/master/LICENSE
[EVIT-files]: https://huggingface.co/api/models/timm/efficientvit_b0.r224_in1k/tree/main
[REP]: https://huggingface.co/timm/repvit_m1_0.dist_450e_in1k
[REP-license]: https://github.com/THU-MIG/RepViT/blob/main/LICENSE
[REP-readme]: https://github.com/THU-MIG/RepViT
[REP-files]: https://huggingface.co/api/models/timm/repvit_m1_0.dist_450e_in1k/tree/main
[HF-zero]: https://huggingface.co/docs/hub/spaces-zerogpu
[HF-static]: https://huggingface.co/docs/hub/spaces-sdks-static
[HF-gpu]: https://huggingface.co/docs/hub/spaces-gpus
[HF-price]: https://huggingface.co/pricing
[HF-terms]: https://huggingface.co/terms-of-service
[HF-overview]: https://huggingface.co/docs/hub/spaces-overview
[HF-tokens]: https://huggingface.co/docs/hub/security-tokens
[HF-download]: https://huggingface.co/docs/huggingface_hub/guides/download
[HF-quota]: https://huggingface.co/docs/huggingface_hub/guides/manage-spaces#track-your-zerogpu-quota
[THREE-license]: https://github.com/mrdoob/three.js/blob/dev/LICENSE
