"""Viser web viewer bridge for IsaacLab eval/play scripts.

Mirrors the IsaacLab robot state of one environment into a shadow MuJoCo model
and renders it through ``mjviser`` for a browser-based 3D view. Adds velocity
arrows (commanded vs actual, linear vs yaw) and an optional joystick that can
override the ``base_velocity`` command term while the eval loop runs.

This is a slim, single-file rewrite of the holosoma-side bridge tailored to
booster_amp_lab's eval/play scripts. There is no terrain handling, no
checkpoint hot-swap, and no reward streaming — just the parts that are useful
for visually inspecting tracking behaviour in real time.

Usage::

    from viser_bridge import BoosterViserBridge

    bridge = BoosterViserBridge(env)  # env is the unwrapped ManagerBasedRLEnv
    cmd_term = env.command_manager.get_term("base_velocity")
    while running:
        if bridge.joystick_enabled:
            cmd_term.vel_command_b[:, 0] = bridge.joystick_command[0]
            cmd_term.vel_command_b[:, 1] = bridge.joystick_command[1]
            cmd_term.vel_command_b[:, 2] = bridge.joystick_command[2]
        ...
        bridge.update()
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from xml.etree import ElementTree as ET

import numpy as np

logger = logging.getLogger("booster_viser_bridge")

_DEFAULT_K1_MJCF = "robots/K1/K1_22dof.xml"
_ARROW_SHAFT_RATIO = 0.8
_ARROW_HEAD_RATIO = 0.2
_ARROW_WIDTH = 0.015
_Z = np.array([0.0, 0.0, 1.0])


def _rotation_between(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Quaternion (wxyz) rotating ``src`` onto ``dst``."""
    c = float(np.dot(src, dst))
    if c > 1.0 - 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1.0 + 1e-8:
        perp = np.array([1.0, 0.0, 0.0]) if abs(src[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(src, perp)
        axis /= np.linalg.norm(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]])
    axis = np.cross(src, dst)
    w = 1.0 + c
    q = np.array([w, axis[0], axis[1], axis[2]])
    return q / np.linalg.norm(q)


def _colored(mesh, rgba):
    mesh.visual.face_colors = [rgba] * len(mesh.faces)
    return mesh


class _Arrow3D:
    """Cylinder shaft + cone head arrow in the viser scene."""

    def __init__(self, server, name: str, rgba: tuple[int, int, int, int]):
        import trimesh

        shaft = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=12)
        shaft.apply_translation([0, 0, 0.5])
        head = trimesh.creation.cone(radius=2.0, height=1.0, sections=12)
        self._shaft = server.scene.add_mesh_trimesh(f"{name}/shaft", _colored(shaft, rgba))
        self._head = server.scene.add_mesh_trimesh(f"{name}/head", _colored(head, rgba))
        self._visible = True

    def update(self, start: np.ndarray, end: np.ndarray, offset: np.ndarray) -> None:
        s, e = start + offset, end + offset
        d = e - s
        length = float(np.linalg.norm(d))
        if length < 1e-4:
            self._shaft.visible = False
            self._head.visible = False
            return
        self._shaft.visible = self._visible
        self._head.visible = self._visible
        if not self._visible:
            return
        direction = d / length
        q = _rotation_between(_Z, direction)
        w = _ARROW_WIDTH
        self._shaft.position = s
        self._shaft.wxyz = q
        self._shaft.scale = (w, w, _ARROW_SHAFT_RATIO * length)
        self._head.position = s + direction * _ARROW_SHAFT_RATIO * length
        self._head.wxyz = q
        self._head.scale = (w, w, _ARROW_HEAD_RATIO * length)

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        self._shaft.visible = visible
        self._head.visible = visible


def _resolve_default_mjcf() -> str:
    """Resolve the K1 MJCF path inside the installed booster_assets package."""
    try:
        from booster_assets import BOOSTER_ASSETS_DIR
    except ImportError as e:
        raise RuntimeError(
            "Could not import booster_assets to resolve default MJCF path. "
            "Install booster_assets or pass mjcf_path= explicitly."
        ) from e
    return os.path.join(BOOSTER_ASSETS_DIR, _DEFAULT_K1_MJCF)


def _load_shadow_model(mjcf_path: str):
    """Load the MJCF, adding an invisible floor so mj_forward succeeds."""
    import mujoco as mj

    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "geom", {
        "name": "_viser_floor", "type": "plane", "size": "10 10 0.01",
        "rgba": "0 0 0 0", "contype": "1", "conaffinity": "1",
    })
    mjcf_dir = os.path.dirname(mjcf_path)
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".xml", dir=mjcf_dir, delete=False) as tmp:
        tree.write(tmp, xml_declaration=True, encoding="utf-8")
        tmp_path = tmp.name
    try:
        model = mj.MjModel.from_xml_path(tmp_path)
    finally:
        os.unlink(tmp_path)
    return model, mj.MjData(model)


class BoosterViserBridge:
    """Viser bridge that mirrors one IsaacLab env into a shadow MuJoCo scene."""

    def __init__(
        self,
        env,
        *,
        env_id: int = 0,
        mjcf_path: str | None = None,
        host: str = "0.0.0.0",
        port: int = 8080,
        update_freq: int = 1,
        fps_limit: int = 60,
    ) -> None:
        # late imports so this module is cheap to import without viser installed
        import mujoco as mj
        import viser as _viser
        from mjviser import ViserMujocoScene

        for noisy in ("websockets", "websockets.server", "trimesh", "trimesh.util"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        self._env = env
        self._env_id = env_id
        self._update_freq = max(1, update_freq)
        self._min_interval = 1.0 / max(fps_limit, 1)
        self._step_count = 0
        self._total_steps = 0
        self._last_update_time = 0.0

        self._mjcf_path = mjcf_path or _resolve_default_mjcf()
        self._mj_model, self._mj_data = _load_shadow_model(self._mjcf_path)
        self._mj = mj
        self._build_joint_addressing()

        self._server = _viser.ViserServer(host=host, port=port)
        self._scene = ViserMujocoScene(self._server, self._mj_model, num_envs=1)

        self._arrow_cmd_lin = _Arrow3D(self._server, "/arrows/cmd_lin", (50, 70, 230, 220))
        self._arrow_cmd_ang = _Arrow3D(self._server, "/arrows/cmd_ang", (50, 150, 50, 220))
        self._arrow_actual_lin = _Arrow3D(self._server, "/arrows/actual_lin", (0, 200, 255, 200))
        self._arrow_actual_ang = _Arrow3D(self._server, "/arrows/actual_ang", (0, 230, 100, 200))
        self._show_arrows = True
        self._arrow_scale = 0.5
        self._arrow_z = 0.2

        self._joystick_enabled = False
        self._joystick_vx = 0.0
        self._joystick_vy = 0.0
        self._joystick_yaw = 0.0
        self._speed_multiplier = 1.0

        self._build_gui()

        logger.info("BoosterViserBridge: http://%s:%d", host, port)
        print(f"[viser] http://{host}:{port}")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_joint_addressing(self) -> None:
        """Map IsaacLab joint order → MuJoCo qpos addresses for hinge joints."""
        mj = self._mj
        model = self._mj_model
        if model.njnt > 0 and model.jnt_type[0] == mj.mjtJoint.mjJNT_FREE:
            self._free_qpos_addr: int | None = int(model.jnt_qposadr[0])
        else:
            self._free_qpos_addr = None

        mj_joint_addr: dict[str, int] = {}
        for jid in range(model.njnt):
            jname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid)
            if jname and model.jnt_type[jid] != mj.mjtJoint.mjJNT_FREE:
                mj_joint_addr[jname] = int(model.jnt_qposadr[jid])

        self._sim_joint_names: list[str] = list(self._env.scene["robot"].data.joint_names)
        self._sim_to_qpos: list[int] = []
        missing: list[str] = []
        for sim_name in self._sim_joint_names:
            addr = mj_joint_addr.get(sim_name)
            if addr is None:
                missing.append(sim_name)
                addr = -1
            self._sim_to_qpos.append(addr)

        if missing:
            logger.warning(
                "BoosterViserBridge: %d joints missing in MJCF (will be left at default): %s",
                len(missing), missing[:5],
            )

    def _build_gui(self) -> None:
        import viser as _viser

        server = self._server
        tabs = server.gui.add_tab_group()

        with tabs.add_tab("Controls"):
            with server.gui.add_folder("Info", expand_by_default=True):
                self._info_md = server.gui.add_markdown(self._info_text())

            with server.gui.add_folder("Simulation", expand_by_default=True):
                btn_pause = server.gui.add_button("Pause", icon=_viser.Icon.PLAYER_PAUSE)
                btn_play = server.gui.add_button("Play", icon=_viser.Icon.PLAYER_PLAY, visible=False)

                @btn_pause.on_click
                def _(_evt):
                    self._scene.paused = True
                    btn_pause.visible = False
                    btn_play.visible = True

                @btn_play.on_click
                def _(_evt):
                    self._scene.paused = False
                    btn_play.visible = False
                    btn_pause.visible = True

                speed = server.gui.add_button_group("Speed", ("Slower", "1x", "Faster"))

                @speed.on_click
                def _(_evt):
                    if speed.value == "Slower":
                        self._speed_multiplier = max(0.125, self._speed_multiplier / 2.0)
                    elif speed.value == "Faster":
                        self._speed_multiplier = min(8.0, self._speed_multiplier * 2.0)
                    else:
                        self._speed_multiplier = 1.0

            with server.gui.add_folder("Velocity arrows", expand_by_default=True):
                cb_show = server.gui.add_checkbox("Show", initial_value=True)
                sl_scale = server.gui.add_slider("Scale", min=0.1, max=3.0, step=0.1, initial_value=0.5)
                sl_z = server.gui.add_slider("Height", min=0.0, max=1.0, step=0.05, initial_value=0.2)

                @cb_show.on_update
                def _(_evt):
                    self._show_arrows = cb_show.value
                    for a in (self._arrow_cmd_lin, self._arrow_cmd_ang,
                              self._arrow_actual_lin, self._arrow_actual_ang):
                        a.set_visible(cb_show.value)

                @sl_scale.on_update
                def _(_evt):
                    self._arrow_scale = sl_scale.value

                @sl_z.on_update
                def _(_evt):
                    self._arrow_z = sl_z.value

                server.gui.add_markdown(
                    "Blue lin / Green yaw — commanded\n\nCyan lin / Light-green yaw — actual"
                )

            with server.gui.add_folder("Commands (joystick)", expand_by_default=True):
                cb_joy = server.gui.add_checkbox("Override env command", initial_value=False)
                sl_vx = server.gui.add_slider("lin_vel_x", min=-2.0, max=2.0, step=0.05, initial_value=0.0)
                sl_vy = server.gui.add_slider("lin_vel_y", min=-1.0, max=1.0, step=0.05, initial_value=0.0)
                sl_yaw = server.gui.add_slider("ang_vel_z", min=-2.0, max=2.0, step=0.05, initial_value=0.0)
                btn_zero = server.gui.add_button("Zero", icon=_viser.Icon.SQUARE_X)

                @cb_joy.on_update
                def _(_evt):
                    self._joystick_enabled = cb_joy.value

                @sl_vx.on_update
                def _(_evt):
                    self._joystick_vx = float(sl_vx.value)

                @sl_vy.on_update
                def _(_evt):
                    self._joystick_vy = float(sl_vy.value)

                @sl_yaw.on_update
                def _(_evt):
                    self._joystick_yaw = float(sl_yaw.value)

                @btn_zero.on_click
                def _(_evt):
                    sl_vx.value = sl_vy.value = sl_yaw.value = 0.0
                    self._joystick_vx = self._joystick_vy = self._joystick_yaw = 0.0

        with tabs.add_tab("Scene"):
            self._scene.create_scene_gui(
                camera_distance=3.0, camera_azimuth=150.0, camera_elevation=25.0,
            )
        with tabs.add_tab("Visualization"):
            self._scene.create_overlay_gui()
        with tabs.add_tab("Groups"):
            self._scene.create_groups_gui()

    def _info_text(self) -> str:
        status = "Paused" if self._scene.paused else "Running"
        return (
            f"**Status:** {status}\n\n"
            f"**Step:** {self._total_steps}\n\n"
            f"**Speed:** {self._speed_multiplier}x\n\n"
            f"**Env id:** {self._env_id}\n\n"
            f"**Joints mapped:** {sum(a >= 0 for a in self._sim_to_qpos)}/{len(self._sim_to_qpos)}"
        )

    # ------------------------------------------------------------------
    # State sync
    # ------------------------------------------------------------------

    def _read_robot_state(self):
        robot = self._env.scene["robot"]
        rs = robot.data.root_state_w[self._env_id].detach().cpu().numpy()
        # IsaacLab order: pos[3], quat_wxyz[4], lin_vel_w[3], ang_vel_w[3]
        pos = rs[0:3]
        quat_wxyz = rs[3:7]
        lin_vel_w = rs[7:10]
        ang_vel_w = rs[10:13]
        joint_pos = robot.data.joint_pos[self._env_id].detach().cpu().numpy()
        return pos, quat_wxyz, lin_vel_w, ang_vel_w, joint_pos

    def _read_command(self):
        try:
            term = self._env.command_manager.get_term("base_velocity")
            cmd = term.vel_command_b[self._env_id].detach().cpu().numpy()
            cmd_vx = float(cmd[0]) if cmd.size > 0 else 0.0
            cmd_vy = float(cmd[1]) if cmd.size > 1 else 0.0
            cmd_yaw = float(cmd[2]) if cmd.size > 2 else 0.0
        except Exception:
            cmd_vx = cmd_vy = cmd_yaw = 0.0
        return cmd_vx, cmd_vy, cmd_yaw

    def _sync_shadow(self, pos, quat_wxyz, joint_pos) -> None:
        data = self._mj_data
        if self._free_qpos_addr is not None:
            a = self._free_qpos_addr
            data.qpos[a:a + 3] = pos
            # MuJoCo and IsaacLab both store free joint quat as (w, x, y, z).
            data.qpos[a + 3:a + 7] = quat_wxyz
        for sim_idx, addr in enumerate(self._sim_to_qpos):
            if addr >= 0:
                data.qpos[addr] = float(joint_pos[sim_idx])
        self._mj.mj_forward(self._mj_model, data)

    def _quat_to_R(self, quat_wxyz: np.ndarray) -> np.ndarray:
        w, x, y, z = quat_wxyz
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    def _scene_offset(self) -> np.ndarray:
        """If mjviser is tracking a body, scene is shifted by -tracked_pos in xy."""
        if not getattr(self._scene, "camera_tracking_enabled", False):
            return np.zeros(3)
        tracked_id = getattr(self._scene, "_tracked_body_id", None)
        if tracked_id is None or tracked_id >= self._mj_model.nbody:
            return np.zeros(3)
        offset = -self._mj_data.xpos[tracked_id].copy()
        offset[2] = 0.0
        return offset

    def _update_arrows(self, pos, R, lin_vel_w, ang_vel_w, cmd) -> None:
        cmd_vx, cmd_vy, cmd_yaw = cmd
        scale = self._arrow_scale
        z_off = self._arrow_z
        offset = self._scene_offset()

        # body → world helper (scale included so lengths render reasonably)
        def b2w(local_vec: np.ndarray) -> np.ndarray:
            return pos + R @ (local_vec * scale)

        base_offset = np.array([0.0, 0.0, z_off])
        origin = b2w(base_offset)

        lin_vel_b = R.T @ lin_vel_w
        ang_vel_b = R.T @ ang_vel_w

        self._arrow_cmd_lin.update(
            origin, b2w(base_offset + np.array([cmd_vx, cmd_vy, 0.0])), offset)
        self._arrow_cmd_ang.update(
            origin, b2w(base_offset + np.array([0.0, 0.0, cmd_yaw])), offset)
        self._arrow_actual_lin.update(
            origin, b2w(base_offset + np.array([lin_vel_b[0], lin_vel_b[1], 0.0])), offset)
        self._arrow_actual_ang.update(
            origin, b2w(base_offset + np.array([0.0, 0.0, ang_vel_b[2]])), offset)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def joystick_enabled(self) -> bool:
        return self._joystick_enabled

    @property
    def joystick_command(self) -> tuple[float, float, float]:
        return self._joystick_vx, self._joystick_vy, self._joystick_yaw

    @property
    def speed_multiplier(self) -> float:
        return self._speed_multiplier

    @property
    def paused(self) -> bool:
        return bool(self._scene.paused)

    def update(self) -> None:
        """Push current env state to the viser scene."""
        self._step_count += 1
        self._total_steps += 1
        if self._step_count % self._update_freq != 0:
            return
        if self._scene.paused:
            return
        now = time.monotonic()
        interval = self._min_interval / max(self._speed_multiplier, 0.1)
        if (now - self._last_update_time) < interval:
            return
        self._last_update_time = now

        try:
            pos, quat_wxyz, lin_vel_w, ang_vel_w, joint_pos = self._read_robot_state()
        except Exception:
            logger.exception("BoosterViserBridge: failed to read robot state")
            return
        cmd = self._read_command()

        self._sync_shadow(pos, quat_wxyz, joint_pos)

        with self._server.atomic():
            self._scene.update_from_mjdata(self._mj_data)
            R = self._quat_to_R(quat_wxyz)
            if self._show_arrows:
                self._update_arrows(pos, R, lin_vel_w, ang_vel_w, cmd)
            if self._total_steps % 30 == 0:
                self._info_md.content = self._info_text()

    def apply_joystick(self, cmd_term) -> None:
        """If joystick is enabled, write its values into the command term.

        Writes to ``cmd_term.vel_command_b`` for *every* env so the override is
        applied uniformly in vectorised eval.
        """
        if not self._joystick_enabled:
            return
        try:
            cmd_term.vel_command_b[:, 0] = self._joystick_vx
            cmd_term.vel_command_b[:, 1] = self._joystick_vy
            if cmd_term.vel_command_b.shape[1] > 2:
                cmd_term.vel_command_b[:, 2] = self._joystick_yaw
        except (AttributeError, IndexError):
            pass

    def close(self) -> None:
        try:
            self._server.stop()
        except Exception:
            try:
                self._server.close()
            except Exception:
                pass
