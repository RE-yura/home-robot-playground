import { Material, Mesh, MeshPhongMaterial, MeshPhysicalMaterial } from "three";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { toCreasedNormals } from "three/addons/utils/BufferGeometryUtils.js";

export class HSROBJLoader extends OBJLoader {
  override parse(source: string) {
    // OBJLoader treats a mixed face/line object entirely as LineSegments. Render its original faces.
    const group = super.parse(source.replace(/^l\s+.*$/gm, ""));
    const materials = new Map<Material, Material>();
    const pbr = (material: Material, part: string) => {
      if (!(material instanceof MeshPhongMaterial)) return material;
      const shell = /^(Arm_|Wrist_Roll)/.test(part);
      const metal = part === "Wrist_Flex_Wrist_Flex", fingers = /^(L|R)_(Proximal|Distal)$/.test(part);
      if (!materials.has(material)) materials.set(material, new MeshPhysicalMaterial({
        name: material.name, color: material.color, map: material.map,
        emissive: material.emissive, emissiveMap: material.emissiveMap,
        opacity: material.opacity, transparent: material.transparent, side: material.side,
        vertexColors: material.vertexColors, flatShading: material.flatShading,
        // Keep MTL colors; distinguish the coated shell, joint metal and matte fingers.
        metalness: metal ? 0.75 : part === "Palm" ? 0.25 : 0,
        roughness: fingers ? 0.78 : shell ? 0.3 : metal ? 0.32 : 0.45,
        clearcoat: shell ? 0.35 : 0, clearcoatRoughness: 0.25,
      }));
      return materials.get(material)!;
    };
    group.traverse(object => {
      if (object instanceof Mesh) {
        if (/^(Arm_|Wrist_)/.test(object.name)) {
          // Repair the lighting normals; keep original positions, UVs and sharp edges.
          // Millimetres give the utility's weld hash a 10 µm tolerance instead of 1 cm.
          const lighting = toCreasedNormals(object.geometry.clone().scale(1000, 1000, 1000));
          object.geometry.setAttribute("normal", lighting.getAttribute("normal"));
          lighting.dispose();
        }
        object.material = Array.isArray(object.material)
          ? object.material.map(material => pbr(material, object.name)) : pbr(object.material, object.name);
      }
    });
    return group;
  }
}
