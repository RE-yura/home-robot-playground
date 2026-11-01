/* eslint-disable react/immutability -- MuJoCo handles and Three.js cameras are mutable external simulation objects. */
import { useEffect, useMemo, useRef, type RefObject } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, useProgress } from "@react-three/drei";
import { PerspectiveCamera, WebGLRenderTarget, SRGBColorSpace } from "three";
import type { Simulation } from "../simulation/mujoco.ts";
import { advancePhysicsFrame } from "./physics-clock.ts";
import { RoomEnvironment } from "./RoomEnvironment.tsx";
import { ManipulationObjects } from "./VisualAssets.tsx";
import { HSRRobot } from "./HSRRobot.tsx";

type VideoView = "head" | "hand" | "overview";

function Scene({ sim, paused, mobile }: { sim: Simulation; paused: boolean | (() => boolean); mobile: boolean }) {
  const clock = useRef({ accumulator: 0, generation: sim.resetGeneration });
  useFrame((_, delta) => {
    const start = performance.now();
    advancePhysicsFrame(sim, delta, paused, clock.current);
    sim.physicsMs = performance.now() - start;
  }, -1);
  return <>
    <ambientLight intensity={0.15}/><hemisphereLight args={["#fff8ef", "#b5a590", 0.35]}/>
    <directionalLight position={[-2, 4, -2.5]} intensity={3.2} color="#fff2dc" castShadow shadow-mapSize={mobile ? [1024, 1024] : [2048, 2048]}
      shadow-camera-left={-4} shadow-camera-right={4} shadow-camera-top={4} shadow-camera-bottom={-4}
      shadow-bias={-0.00015} shadow-normalBias={0.005} shadow-radius={3}/>
    <RoomEnvironment sim={sim}/><HSRRobot sim={sim}/><ManipulationObjects sim={sim}/>
  </>;
}

function Views({ sim, controlView, headPane, handPane, overviewPane, onReady }: { sim: Simulation; controlView: "head" | "hand";
  headPane: RefObject<HTMLDivElement | null>; handPane: RefObject<HTMLDivElement | null>; overviewPane: HTMLDivElement | null;
  onReady: (view: VideoView) => void }) {
  const rendered = useRef(new Set<VideoView>());
  const lastCapture = useRef(-Infinity);
  const capture = useMemo(() => {
    const target = new WebGLRenderTarget(320, 240); target.texture.colorSpace = SRGBColorSpace;
    return { target, pixels: new Uint8Array(320 * 240 * 4), canvas: document.createElement("canvas") };
  }, []);
  useEffect(() => () => capture.target.dispose(), [capture]);
  const set = useThree(s => s.set), head = useMemo(() => new PerspectiveCamera(64, 1, 0.035, 30), []);
  // A 5 mm near plane preserves objects held just centimetres from the palm camera.
  const hand = useMemo(() => new PerspectiveCamera(64, 1, 0.005, 30), []);
  const overview = useMemo(() => {
    const c = new PerspectiveCamera(55, 1, 0.05, 30); c.position.set(2.025, 6.55, 1.6125); c.lookAt(0, 0.7, -0.15); return c;
  }, []);
  useEffect(() => {
    for (const key of Array.from(sim.keys)) sim.setKey(key, false);
    sim.controlView = controlView;
    set({ camera: { head, hand }[controlView] });
  }, [set, sim, controlView, head, hand]);
  useFrame(({ gl, scene, size }) => {
    const learningDue = sim.learningCapture?.needsImages === false ? false : sim.learningCapture?.captureDue(sim.data.time);
    for (const [camera, id] of [[head, sim.cameraId], [hand, sim.handCameraId]] as const) {
      const k = id * 3, p = sim.data.site_xpos;
      camera.position.set(p[k], p[k + 1], p[k + 2]);
      camera.quaternion.fromArray(sim.siteRotation(id));
    }
    sim.overviewRotation = overview.quaternion.toArray();
    const oldClear = gl.autoClear; gl.autoClear = false; gl.setScissorTest(false);
    if (!sim.imageCaptureEnabled) lastCapture.current = -Infinity;
    if (!sim.replaying && (learningDue || (sim.imageCaptureEnabled && (sim.cameraImages.length !== 2 || sim.data.time < lastCapture.current || sim.data.time - lastCapture.current >= 1)))) {
      capture.canvas.width = 320; capture.canvas.height = 240;
      const context = capture.canvas.getContext("2d")!;
      const images = ([ ["head", head], ["hand", hand] ] as const).map(([view, camera]) => {
        camera.aspect = 4 / 3;
        if (camera === hand) camera.fov = 2 * Math.atan(Math.tan(1) / camera.aspect) * 180 / Math.PI;
        camera.updateProjectionMatrix(); gl.setRenderTarget(capture.target); gl.setViewport(0, 0, 320, 240);
        gl.clear(); gl.render(scene, camera); gl.readRenderTargetPixels(capture.target, 0, 0, 320, 240, capture.pixels);
        const frame = context.createImageData(320, 240);
        for (let y = 0; y < 240; y++) frame.data.set(capture.pixels.subarray(y * 1280, (y + 1) * 1280), (239 - y) * 1280);
        context.putImageData(frame, 0, 0);
        return { view, timestamp: sim.data.time, width: 320, height: 240, dataUrl: capture.canvas.toDataURL("image/jpeg", 0.75) };
      });
      const acquisition = sim.publishCameraImages(images);
      lastCapture.current = sim.data.time; gl.setRenderTarget(null);
      if (learningDue) sim.learningCapture?.capture(sim.cameraImages, acquisition);
    }
    if (sim.learningCapture?.inferencePending && sim.learningCapture.pauseDisplayDuringInference !== false) {
      gl.setViewport(0, 0, size.width, size.height); gl.setScissorTest(false); gl.autoClear = oldClear;
      return;
    }
    gl.setViewport(0, 0, size.width, size.height); gl.clear(); gl.setScissorTest(true);
    const canvas = gl.domElement.getBoundingClientRect();
    for (const [view, camera, pane] of [["head", head, headPane.current], ["hand", hand, handPane.current], ["overview", overview, overviewPane]] as const) {
      const rect = pane?.getBoundingClientRect();
      if (!rect || rect.width <= 0 || rect.height <= 0) continue;
      camera.aspect = rect.width / rect.height;
      // HSR's published hand sensor has a 2 rad horizontal field of view.
      if (camera === hand) camera.fov = 2 * Math.atan(Math.tan(1) / camera.aspect) * 180 / Math.PI;
      camera.updateProjectionMatrix();
      const x = rect.left - canvas.left, y = size.height - (rect.bottom - canvas.top);
      gl.setViewport(x, y, rect.width, rect.height); gl.setScissor(x, y, rect.width, rect.height);
      gl.clear(); gl.render(scene, camera);
      if (!rendered.current.has(view)) { rendered.current.add(view); onReady(view); }
    }
    gl.setViewport(0, 0, size.width, size.height); gl.setScissorTest(false); gl.autoClear = oldClear;
  }, 1);
  return overviewPane && <OrbitControls camera={overview} domElement={overviewPane} target={[0, 0.7, -0.15]}
    minDistance={1.3} maxDistance={10} maxPolarAngle={Math.PI / 2 - 0.03}/>;
}

function AssetErrors() {
  const errors = useProgress(state => state.errors);
  if (!errors.length) return null;
  return <div className="viewport-error" role="alert">
    部屋の素材を読み込めませんでした。ページを再読み込みしてください。
  </div>;
}

// active=false (the other menu is shown) stops the frame loop: no rendering and no physics until shown again.
export default function SimulationCanvas({ sim, active = true, paused, mobile, controlView, headPane, handPane, overviewPane, onReady }: {
  sim: Simulation | null; active?: boolean; paused: boolean | (() => boolean); mobile: boolean; controlView: "head" | "hand";
  headPane: RefObject<HTMLDivElement | null>; handPane: RefObject<HTMLDivElement | null>; overviewPane: HTMLDivElement | null;
  onReady: (view: VideoView) => void;
}) {
  return <>
    <div className="simulation-canvas">
      {sim ? <Canvas frameloop={active ? "always" : "never"} shadows dpr={mobile ? 1 : [1, 1.5]} camera={{ position: [2.5, 2.2, 3] }}>
        <color attach="background" args={["#e8e5df"]}/><Scene sim={sim} paused={paused} mobile={mobile}/>
        <Views sim={sim} controlView={controlView} headPane={headPane} handPane={handPane} overviewPane={overviewPane} onReady={onReady}/>
      </Canvas> : null}
    </div>
    <AssetErrors/>
  </>;
}
