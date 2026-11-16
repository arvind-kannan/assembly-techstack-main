# AT-249 build plan (session 1, orientation only, no code yet)

## Facts gathered

- Hao repo cloned bare at `C:\Users\ve2796\.claude\jobs\db568c38\tmp\hao` (may be gone; re-clone with
  `git -c "credential.helper=!glab auth git-credential" clone --no-checkout https://gitlab.kit.edu/kit/wbk/map/imk/sfb1574_kreislauffabrik/MA_Hao_perception_MR_repositioning.git`),
  read files with `git show f51a3f6:src/ma_pipeline/perception/<file>`.
- No blockers. The ticket is assigned to the bot (id 25384).
- The venv works: `uv sync --group dev` done in the worktree. Open3D 0.20.0. **cv2 and pybullet are missing.** Add `uv add opencv-python-headless` (needs `cv2.aruco.ArucoDetector`, OpenCV >= 4.7). Do not pin Open3D below 0.20.
- The desk mesh is `crc_disassembly_sim/fixtures/Spanntisch/spanntisch_low_res.stl`, in mm (the URDF scales it by 0.001). Bounds are x,y -95..1105 and top z = 300. This is the same frame as Hao's `platform_root`, so his model applies: top bounds [-0.095,-0.095]..[1.105,1.105] m, top z 0.300, support centre (0.505,0.505,0.300).
- Hole catalogue: the CSV sits in Hao's uncommitted `logs/`. **Generate it instead:** a 12x12 base grid at 100 mm centred on (0.505,0.505), so x,y = -0.045..1.055, plus an 11x11 diagonal grid at 0.005..1.005. That gives 144 + 121 = 265 holes, with `pattern_type` `base_grid` / `diagonal`. Verify it by raycasting the desk STL: at a hole, a ray misses or hits well below z = 0.300.
- Clamp meshes, all in mm:
  - `crc_disassembly_sim/fixtures/Large_system/large_system_body.stl` (+ fingers)
  - `crc_disassembly_sim/fixtures/Small_system/body.stl`

  `Objects/fixture_mesh/Fix01_Base.obj` is a generic 300x400 base in metres. Do not use it as the clamp.
- Legacy #45 gate:
  - Code: `perceptionModule.check_icp_match`: `fitness > 0.7 and inlier_rmse < 1.0 and fitness/inlier_rmse > 1.5`, all in **mm**.
  - Legacy registration (`Registration.py`): RANSAC on a coarse voxel (1 mm), then ICP on a 0.25 mm voxel with threshold 0.75 mm, point-to-point.
  - Source = the sim-rendered visible part segment; target = the real cloud.
- Hao's profile values are in `configs/stage92jc_angle_grinder_capture_agnostic_v1.yaml` (`registration`, `hard_gates`, `platform_localization.hole_detection`/`grid_fit`, yaw branches 0/90/180/270).

## Design (advisor-checked)

One new package outside the subtrees: `src/perception/` (importable via pytest.ini `src`).

1. `zivid_capture.py`:
   - `ZividCapture` with a lazy `import zivid`.
   - `frame_to_organized(frame)` returns an organized cloud (`xyz_m` HxWx3, `rgb` HxWx3 uint8).
   - **mm to m exactly once**, in one function.
   - `load_zdf`.
   - Test with a duck-typed fake frame (`point_cloud().copy_data("xyz"/"rgba")`).
   - Add an adapter for a DisassemblySim `RGBDCamera.last_frame`:
     - Rebuild the organized cloud from `depth_img` + the projection matrix.
     - Filter on `mask`, because `cutoff_depth` compares against the 0-1 depth buffer.
     - OpenGL eye frame to the Zivid convention = `flip_yz`.
2. `registration.py`: Hao's `calculate_registration` (FPFH + RANSAC + ICP), seeded, returning a dataclass.
3. `quality_gate.py`:
   - `LegacyIcpGate` defaults: min_fitness 0.7, max_rmse_m 0.001, **ratio = fitness / (rmse_m*1000) > 1.5**.
   - `PlausibilityGate`: 5 mm / 3 deg against the nominal pose.
   - Hao's `evaluate_global_hard_gates` as opt-in extras. Drop the output-face and table-support gates, or run them only when their inputs are given.
4. `registrar.py`: a generic model-to-observation registrar, used for the product and the clamps.
   - Coarse multi-seed FPFH/RANSAC (Hao), plus an optional nominal-pose seed for #45.
   - Robust point-to-plane ICP.
   - **Fine ICP for the gate metrics** on dense clouds (0.75 mm threshold).
   - Gate fitness computed as `evaluate_registration(visible model subset, observation, thr, T)`. The visible subset comes from hidden-point removal from the camera origin, as in the legacy visible-segment semantics. First write a 10-line toy test to confirm the Open3D fitness denominator (source points).
   - Dedup by Hao's rule (15 mm / 12 deg).
   - Status: UNIQUE_PASS / AMBIGUOUS / REJECTED.
   - **Never loosen the defaults to get green.**
5. `system28.py`: desk constants, the generated hole catalogue, and loading the desk mesh (mm to m once).
6. `platform_localizer.py`:
   - Hao's `localize_system28` (Hough circles, ray-to-plane, grid fit, 4 yaw hypotheses) and `fit_projection_pinhole`.
   - A table-plane fit via Open3D `segment_plane`.
   - Allow a hole image built from plane membership (holes dark) as well as RGB, so that sim renders work.
   - Use `depth_adaptive` or Zivid-like intrinsics for the Hough radii.
7. `platform_markers.py`: Hao's white-shape C4 selector, plus a thin `cv2.aruco` DICT_4X4_50 selector with the same interface. Fallback is the nearest-to-nominal yaw (park 046f).
8. `clamps.py`: crop the cloud above the desk top and register each clamp mesh with the registrar, seeded by nominal desk-frame poses if given (park d053).
9. `pose_provider.py`: `(cloud_camera, T_world_camera, product model, desk/clamp specs, nominals)` returns `T_world_product`, `T_world_desk` and `T_world_clamp[name]`, plus a gate report per pose. This is #45's import surface.
10. M60 settings YAML derived from the M70 one, marked unvalidated (park 77ea). Put it next to the package, e.g. `src/perception/config/`.
11. `zdf_product_geometry.py`: port only the decode, the exactly-once scaling and the plane segmentation above the dominant plane.

Tests: under `tests/unit/perception/`. Use an Open3D `RaycastingScene` pinhole renderer of the sim meshes (desk + clamp + one product) that produces an organized xyz and an intensity image. Cover:

- a pass case for the product and the desk;
- a reject case at the gate (for example a wrong model or a shifted nominal beyond 5 mm);
- a units test;
- a hole-catalogue-vs-STL test.

Keep the CI tests free of pybullet, the Zivid SDK and `.zdf` files, because diff-cover needs 60 % of the changed `src/` lines. Mark heavy tests `slow`/`geom` only if they exceed a few seconds.

Docs: one line in CONTEXT-MAP (perception code location) or in the motion-planning context. ADR-0011 says legacy perception stays until #171.

## MR description lines (interim)

- Clamp localisation and the ArUco yaw selector built new (not in Hao's code): decided without Max Goebels (Maintainer) (park 20261021-004038-d053)
- Desk yaw without markers picked nearest to the nominal desk pose: decided without Max Goebels (Maintainer) (park 20261021-004038-046f)
- M60 settings file derived from the M70 one, unvalidated on hardware: decided without Max Goebels (Maintainer) (park 20261021-004039-77ea)
