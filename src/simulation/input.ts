export const ARM_KEYS = ["KeyI", "KeyK", "KeyJ", "KeyL", "KeyU", "KeyO"];
export const BASE_KEYS = ["KeyW", "KeyS", "KeyA", "KeyD", "KeyQ", "KeyE"];
export const SHIFT_KEYS = ["ShiftLeft", "ShiftRight"];
export const CONTROL_KEYS = [...BASE_KEYS, ...ARM_KEYS, ...SHIFT_KEYS, "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Space"];
export type CameraView = "head" | "hand" | "overview";
export type TouchInput = { forward: number; strafe: number; turn: number; lift: number; reach: number };
export const TOUCH_STOP: TouchInput = { forward: 0, strafe: 0, turn: 0, lift: 0, reach: 0 };

