# mlx-kinematics — project plan

**Goal:** a pip-installable robot kinematics library for Apple's MLX. Load a robot arm's URDF, compute where every link is for thousands of joint configurations at once, get Jacobians, and solve inverse kinematics on the Mac GPU.

**Prior art (checked Oct 2026):**
- [pytorch_kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics) (~800 stars): the PyTorch equivalent; only mentions CUDA as its GPU backend.
- [curobo-mlx](https://github.com/RobotFlow-Labs/curobo-mlx) (1 star, 15 commits): a port of NVIDIA's cuRobo motion planner to MLX, with FK, IK, collision and trajectory optimization. It installs from a clone with cuRobo as a submodule, isn't on PyPI, and its README lists no Jacobian API and no head-to-head benchmarks.

**Your angle:** a small, standalone, pip-installable kinematics library (FK, Jacobians, IK) that is tested against a reference and benchmarked head to head against pytorch_kinematics and curobo-mlx. Small and verified beats big and unproven.

**Headline for the README at the end:** a `pip install`, a speed chart against NumPy, pytorch_kinematics and curobo-mlx, and a GIF of an arm following your mouse.

**Pace:** chill, one milestone per weekend-ish. All Python, all on your Mac.

---

## What you'll learn

- MLX: arrays, lazy evaluation, `grad`, `vmap`, `compile`
- Robot kinematics: 4×4 transforms, URDF files, forward kinematics, Jacobians, inverse kinematics
- Packaging and testing a real Python library

---

## Setup

- Python 3.11+ in a venv; `pip install mlx numpy pytest`
- For checking your answers later: `pip install pytorch-kinematics` (and `torch`)
- A robot model: a 7-joint arm like the Franka Panda or KUKA iiwa. URDFs are available through the `robot_descriptions` Python package, and MuJoCo Menagerie has many arms (mostly in MJCF format).

---

## Milestones

### M0 — Learn MLX (an afternoon or two)
- [ ] Redo a few NumPy snippets in `mlx.core`
- [ ] Lazy evaluation: see that nothing runs until `mx.eval()` or a print
- [ ] `mx.grad` on a scalar function, `mx.vmap` to batch a one-example function
- [ ] `mx.compile` on a small function and time it before and after
- [ ] Learn to time correctly: call `mx.eval()` inside the timed region

**Done when:** you can batch and differentiate a small function and time it honestly.

### M1 — Robot math in plain NumPy
- [ ] 4×4 homogeneous transforms: rotation + translation, composing, inverting
- [ ] Rotation about an axis by an angle (revolute joint)
- [ ] Forward kinematics for a 2-link planar arm, checked by hand on paper
- [ ] Generalize to any serial chain of joints

**Done when:** your 2-link arm gives the same answer as your hand calculation.

### M2 — Load a real robot
- [ ] Parse a URDF: links, joints (origin `xyz` + `rpy`, `axis`, `type`, `limit`)
- [ ] Build the chain from base to end effector
- [ ] Support revolute, prismatic and fixed joints
- [ ] Check your FK against pytorch_kinematics for 100 random configurations (error < 1e-5)

**Done when:** your FK matches the reference on a real 7-joint arm.

### M3 — Port to MLX and batch it
- [ ] Input: joint angles of shape (B, n_joints); output: link poses (B, n_links, 4, 4)
- [ ] Try native batching vs `mx.vmap`; keep the faster one
- [ ] Matches NumPy within float32 tolerance

**Done when:** 10,000 configurations in one call, correct.

### M4 — Jacobians
- [ ] Analytic geometric Jacobian (each joint's axis and position)
- [ ] Autodiff Jacobian built from `mx.vjp` / `mx.jvp` + `mx.vmap`
- [ ] Check that the two agree; benchmark both

**Done when:** (B, 6, n_joints) Jacobians, two ways, agreeing.

### M5 — Inverse kinematics
- [ ] Damped least squares, batched over targets
- [ ] Random restarts, keep the best solution per target
- [ ] Respect joint limits
- [ ] Report success rate, position/orientation error and time per solve

**Done when:** it reliably reaches reachable targets and you can quote the numbers.

### M6 — Benchmarks
- [ ] Contenders: NumPy, pytorch_kinematics (CPU and MPS), curobo-mlx, your MLX version, MLX + `mx.compile`
- [ ] Batch sizes from 1 to 100,000; FK, Jacobian and IK
- [ ] Chart: time vs batch size, log scale

**Done when:** you have a chart and an honest sentence about where MLX wins and where it doesn't.

### M7 — Make it a real package
- [ ] `pyproject.toml`, `pip install -e .`
- [ ] pytest suite: FK vs reference, Jacobian vs finite differences, IK round trip
- [ ] Clean API (below), README with install + 10-line example
- [ ] Check the name is free on PyPI, then publish

### M8 — Demo
- [ ] Arm follows your mouse in a simple viewer (matplotlib 3D, or the MuJoCo viewer)
- [ ] Record a GIF for the README

---

## API sketch (change it as you like)

```python
import mlx_kinematics as mk

robot = mk.load_urdf("panda.urdf", end_link="panda_hand")
poses = robot.fk(q)                           # q: (B, 7) -> (B, n_links, 4, 4)
J     = robot.jacobian(q)                     # (B, 6, 7)
q_sol, ok = robot.ik(targets, restarts=32)    # targets: (B, 4, 4)
```

---

## Common traps

- Timing MLX without `mx.eval()` measures nothing (lazy evaluation)
- Check which dtypes MLX supports on the GPU; plan for float32
- URDF `rpy` is roll-pitch-yaw about fixed axes; get the multiplication order right
- Angle wrapping and joint limits in IK

---

## Stretch ideas

- MJCF loading so all of MuJoCo Menagerie works
- Collision spheres + a collision cost in IK
- Write the FK hot loop as a custom Metal kernel with `mx.fast.metal_kernel` (bridge from `metal-matmul`)

---

## Resources

- [MLX documentation](https://ml-explore.github.io/mlx/)
- [Modern Robotics (Lynch & Park), free book + videos](https://hades.mech.northwestern.edu/index.php/Modern_Robotics) — Ch. 3 transforms, Ch. 4 forward kinematics, Ch. 5 Jacobians, Ch. 6 inverse kinematics
- [pytorch_kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics) — reference to check against and to beat
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — robot models
