/* eslint-disable react/immutability -- Three.js transforms belong to the renderer. */
import { Suspense, useEffect, useMemo, useRef, type ComponentProps } from "react";
import { useFrame } from "@react-three/fiber";
import { useGLTF, useTexture } from "@react-three/drei";
import { Box3, Matrix4, Mesh, RepeatWrapping, SRGBColorSpace, Vector3, type Group } from "three";
import type { Simulation } from "../simulation/mujoco.ts";
import { OBJECTS } from "../simulation/scene.ts";
import type { Vec3 } from "../simulation/types.ts";

// All assets are local. Clones share immutable geometry / textures, but not transforms.
export function AssetModel(props: ComponentProps<typeof LoadedAssetModel>) {
  return <Suspense fallback={null}><LoadedAssetModel {...props}/></Suspense>;
}

function LoadedAssetModel({ url, position, rotation = 0, size, origin = "floor" }:
  { url: string; position?: Vec3; rotation?: number; size?: Vec3; origin?: "floor" | "center" | "native" }) {
  const { scene } = useGLTF(url);
  const [width, height, depth] = size ?? [0, 0, 0], fit = Boolean(size);
  const model = useMemo(() => {
    const clone = scene.clone(true);
    clone.traverse(o => { if (o instanceof Mesh) { o.castShadow = true; o.receiveShadow = true; } });
    if (origin !== "native") {
      const bounds = new Box3().setFromObject(clone), dimensions = bounds.getSize(new Vector3());
      if (fit) clone.scale.set(width / dimensions.x, height / dimensions.y, depth / dimensions.z);
      const fitted = new Box3().setFromObject(clone), center = fitted.getCenter(new Vector3());
      clone.position.set(-center.x, origin === "floor" ? -fitted.min.y : -center.y, -center.z);
    }
    return clone;
  }, [scene, width, height, depth, fit, origin]);
  return <group position={position} rotation={[0, rotation, 0]}><primitive object={model}/></group>;
}

export function PbrMaterial(props: ComponentProps<typeof TexturedMaterial>) {
  return <Suspense fallback={<meshStandardMaterial color={props.color ?? "#ffffff"} roughness={props.roughness ?? 1}/>}>
    <TexturedMaterial {...props}/>
  </Suspense>;
}

function TexturedMaterial({ asset, repeat = [1, 1], roughness = 1, color = "#ffffff", painted = false }:
  { asset: string; repeat?: [number, number]; roughness?: number; color?: string; painted?: boolean }) {
  const textures = useTexture(["color", "normal", "roughness"].map(map => `/assets/realistic-home-v1/materials/${asset}/${map}.jpg`));
  const [repeatX, repeatY] = repeat;
  const maps = useMemo(() => textures.map((t, i) => {
    const clone = t.clone(); clone.wrapS = clone.wrapT = RepeatWrapping;
    clone.repeat.set(repeatX, repeatY); clone.anisotropy = 8;
    if (i === 0) clone.colorSpace = SRGBColorSpace;
    clone.needsUpdate = true; return clone;
  }), [textures, repeatX, repeatY]);
  useEffect(() => () => maps.forEach(map => map.dispose()), [maps]);
  return <meshStandardMaterial map={painted ? undefined : maps[0]} normalMap={maps[1]} normalScale={painted ? [0.12, 0.12] : [0.35, 0.35]}
    roughnessMap={maps[2]} roughness={roughness} color={color}/>;
}

export function ManipulationObjects({ sim }: { sim: Simulation }) {
  const groups = useRef<(Group | null)[]>([]), matrix = useMemo(() => new Matrix4(), []);
  const bodies = useMemo(() => OBJECTS.map(({ id }) => sim.id("BODY", id)), [sim]);
  useFrame(() => {
    const p = sim.data.xpos, r = sim.data.xmat;
    groups.current.forEach((group, i) => {
      if (!group) return;
      const body = bodies[i], k = body * 9;
      group.position.set(p[body * 3], p[body * 3 + 1], p[body * 3 + 2]);
      matrix.set(r[k], r[k + 1], r[k + 2], 0, r[k + 3], r[k + 4], r[k + 5], 0,
        r[k + 6], r[k + 7], r[k + 8], 0, 0, 0, 0, 1);
      group.quaternion.setFromRotationMatrix(matrix);
    });
  }, -0.5);
  return <>{OBJECTS.map(({ id, type }, i) => <group key={id} name={`object:${id}`} ref={g => { groups.current[i] = g; }}>
    <AssetModel url={type === "plate" ? "/assets/realistic-home-v1/models/carved_wooden_plate/model.gltf" : `/assets/realistic-home-v1/generated/${type === "ball" ? "ball-v2" : type}.gltf`}
      origin={type === "plate" ? "center" : "native"} size={type === "plate" ? [0.21, 0.027, 0.21] : undefined}
      position={type === "plate" ? [0, -0.0005, 0] : undefined}/>
  </group>)}</>;
}
