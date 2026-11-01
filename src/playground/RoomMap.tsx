import type { Observation, ObjectType } from "../simulation/types.ts";
import { MOVABLE_FURNITURE, FURNITURE, FRIDGE_DOOR } from "../simulation/environment.ts";

const COLORS: Record<ObjectType, string> = {
  cube: "#4e7183", ball: "#c2d139", baseball: "#ebe5d2", can: "#d5ac49", mug: "#416796", bottle: "#366954", plate: "#a9b9ae",
  tea: "#799459", juice: "#d39c47", milk: "#7499b4",
};

// Room coordinates: +X is right and -Z is up. MuJoCo +yaw turns toward -X.
export function RoomMap({ observation, headPan, doorAngle = 0 }: { observation: Observation | null; headPan: number; doorAngle?: number }) {
  const robot = observation?.robot, p = robot?.base.position ?? [0, 0, 1.25], yaw = robot?.base.yaw ?? 0;
  const degrees = yaw * 180 / Math.PI;
  const minX = Math.min(-3.65, p[0] - 0.6), minZ = Math.min(-3.45, p[2] - 0.6);
  const maxX = Math.max(3.65, p[0] + 0.6), maxZ = Math.max(3.05, p[2] + 0.6);
  return <div className="room-map">
    <svg viewBox={`${minX} ${minZ} ${maxX - minX} ${maxZ - minZ}`} role="img" aria-label="部屋の Map。三角が台車の前方、黄色の扇が頭カメラの向き。">
      <rect x="-3.3" y="-3.15" width="6.6" height="5.9" rx="0.08" fill="#ece6d9"/>
      <path d="M-3.3 2.75V-3.15H3.3V2.75Z" fill="none" stroke="#86958c" strokeWidth="0.09"/>
      <g fill="#c7b398" stroke="#ad9980" strokeWidth="0.035">
        <rect x="-1.9" y="-3.0" width="2" height="0.58"/>
        <rect x="0.36" y="-3.08" width="1.53" height="0.75"/>
        <rect x="2.13" y="-3.06" width="0.88" height="0.84" fill="#c8d3cd"/>
        <rect x="-3.12" y="-2.95" width="0.9" height="0.48"/>
        <rect x="-3.12" y="-1.53" width="0.82" height="1.8" fill="#94a693"/>
        <rect x="-1.97" y="-1.03" width="0.42" height="0.85"/>
        <rect x="-0.8" y="-0.5" width="1.6" height="1" rx="0.035" fill="#c69b70"/>
        {MOVABLE_FURNITURE.map(f => {
          const body = observation?.furniture.find(o => o.id === f.id), position = body?.position ?? f.position;
          const r = body?.rotation;
          const yaw = r ? Math.atan2(2 * (r[0] * r[2] + r[3] * r[1]), 1 - 2 * (r[0] * r[0] + r[1] * r[1])) : f.rotation ?? 0;
          const x = Math.min(...f.boxes.map(b => b[0] - b[3])), z = Math.min(...f.boxes.map(b => b[2] - b[5]));
          const width = Math.max(...f.boxes.map(b => b[0] + b[3])) - x, depth = Math.max(...f.boxes.map(b => b[2] + b[5])) - z;
          return <rect key={f.id} data-testid={`map-${f.id}`} x={x} y={z} width={width} height={depth}
            transform={`translate(${position[0]} ${position[2]}) rotate(${-yaw * 180 / Math.PI})`}
            fill={f.id === "basket" || f.id === "tv_plant" ? "#789589" : f.id === "armchair" ? "#94a693" : "#c7b398"}><title>{f.id === "basket" ? "かご" : f.id === "tv_plant" ? "観葉植物" : f.id === "armchair" ? "肘掛け椅子" : "食卓の椅子"}</title></rect>;
        })}
        <rect x="2.71" y="-0.77" width="0.5" height="1.6"/>
      </g>
      <g transform={`translate(${FURNITURE.find(f => f.id === "fridge")!.position[0] + FRIDGE_DOOR.hinge[0]} ${FURNITURE.find(f => f.id === "fridge")!.position[2] + FRIDGE_DOOR.hinge[2]}) rotate(${-doorAngle * 180 / Math.PI})`}>
        <path data-testid="map-fridge-door" d="M0 0H-0.82" stroke="#667c70" strokeWidth="0.045"/>
      </g>
      <g fill="#667c70" fontSize="0.26" textAnchor="middle">
        <text x="-2.71" y="-0.5">ソファ</text><text x="2.57" y="-2.55">冷蔵庫</text>
        <text x="0" y="0.12">机</text>
        <text x="2.96" y="0.12" transform="rotate(90 2.96 0.12)">TV</text>
      </g>
      {observation?.objects.map(o => <circle key={o.id} cx={o.position[0]} cy={o.position[2]}
        r={o.type === "plate" ? 0.1 : o.type === "ball" ? 0.0335 : o.type === "baseball" ? 0.0365 : 0.065} fill={COLORS[o.type]} stroke="#f9faf4" strokeWidth="0.022"><title>{o.id}</title></circle>)}
      <g transform={`translate(${p[0]} ${p[2]})`}>
        <g data-testid="map-camera" transform={`rotate(${-(degrees + headPan * 180 / Math.PI)})`}>
          <path d="M0 0L-0.56 -0.9A1.06 1.06 0 0 1 0.56 -0.9Z" fill="#d6ae4a" fillOpacity="0.25"/>
          <path d="M0 0V-1.06" stroke="#ae842c" strokeWidth="0.045" strokeDasharray="0.08 0.06"/>
        </g>
        <circle r="0.215" fill="#245944" stroke="#fafbf3" strokeWidth="0.04"/>
        <path data-testid="map-heading" transform={`rotate(${-degrees})`} d="M0 -0.43L0.15 0.11L0 -0.01L-0.15 0.11Z" fill="#fffdf5"/>
      </g>
    </svg>
    <div className="map-info">
      <div className="map-position" data-testid="map-pose">
        x {p[0].toFixed(2)} / z {p[2].toFixed(2)} m · 向き {((degrees % 360 + 360) % 360).toFixed(0)}°
      </div>
      <div className="map-legend">▲ 台車の前方 <i/> 頭カメラの向き</div>
    </div>
  </div>;
}
