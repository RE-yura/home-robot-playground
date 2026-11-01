import type { SixValues } from "./recording.ts";
const EPS = 1e-8;
export type RtcTicket = { id: number; index: number; timestamp: number; prefix: SixValues[]; hold?: SixValues };
function checked(actions: number[][]): SixValues[] {
  if (actions.length !== 16 || actions.some(a => a.length !== 6 || !a.every(Number.isFinite))) throw new Error("RTC needs sixteen finite six-channel goals");
  return actions.map(a => [...a] as SixValues);
}

// Physical-time queue. A returned prefix already issued to the actuator cannot
// be replayed, replaced mid-command or used to hide an exhausted prediction.
// A model trained with an action offset predicts goals starting `offset` steps after its capture, so a reply from the
// capture at index c fills c+offset..c+offset+15 and the queued goals before c+offset keep running.
export class RtcActionQueue {
  readonly anchor: number;
  readonly offset: number;
  private goals = new Map<number, SixValues>();
  private pending: RtcTicket | null = null;
  private nextId = 0;
  private applied = -1;
  private end = 0;
  private stopped = false;
  constructor(anchor: number, offset = 0) {
    if (!Number.isFinite(anchor) || anchor < 0) throw new Error("Invalid RTC clock anchor");
    if (!Number.isInteger(offset) || offset < 0 || offset > 15) throw new Error("Invalid RTC action offset");
    this.anchor = anchor; this.offset = offset;
  }
  // `hold` (e.g. the measured state) fills the first offset goals, which an offset model does not predict.
  prime(actions: number[][], hold?: number[]) {
    if (this.end || this.stopped) throw new Error("RTC queue already initialized or stopped");
    if (this.offset && (!hold || hold.length !== 6 || !hold.every(Number.isFinite))) throw new Error("An offset RTC queue needs a finite hold goal to start");
    const values = checked(actions);
    for (let i = 0; i < this.offset; i++) this.goals.set(i, [...hold!] as SixValues);
    values.forEach((action, i) => this.goals.set(this.offset + i, action)); this.end = this.offset + 16;
  }
  private cursor(time: number) {
    if (!Number.isFinite(time) || time < this.anchor - EPS) throw new Error("Invalid RTC physical time");
    return (time - this.anchor) * 10;
  }
  at(time: number): SixValues {
    const index = Math.floor(this.cursor(time) + EPS), action = this.goals.get(index);
    if (this.stopped || !action) throw new Error("RTC queue underflow");
    if (index < this.applied) throw new Error("RTC physical clock moved backwards");
    this.applied = index;
    for (const old of this.goals.keys()) if (old < index) this.goals.delete(old);
    return [...action];
  }
  has(time: number) { return !this.stopped && this.goals.has(Math.floor(this.cursor(time) + EPS)); }
  // atEnd permits a capture exactly when the queued goals are used up (the caller holds physics until the reply).
  begin(time: number, atEnd = false): RtcTicket {
    if (this.pending) throw new Error("RTC inference already pending");
    const cursor = this.cursor(time), index = Math.round(cursor);
    if (Math.abs(cursor - index) > EPS) throw new Error("RTC capture must be on a100ms boundary");
    if (this.stopped || !(this.goals.has(index) || (atEnd && index === this.end))) throw new Error("RTC queue underflow");
    const prefix: SixValues[] = [];
    for (let i = index; i < this.end; i++) {
      const action = this.goals.get(i); if (!action) throw new Error("RTC queue has a gap");
      prefix.push([...action]);
    }
    // With physics held at the queue end, an offset reply starts offset steps later: the last goal is held until then.
    const last = atEnd && index === this.end && this.offset ? this.goals.get(index - 1) : undefined;
    if (atEnd && index === this.end && this.offset && !last) throw new Error("RTC queue has no goal to hold");
    this.pending = { id: ++this.nextId, index, timestamp: time, prefix, ...(last ? { hold: [...last] as SixValues } : {}) };
    return { ...this.pending, prefix: prefix.map(a => [...a]), ...(last ? { hold: [...last] as SixValues } : {}) };
  }
  accept(ticket: RtcTicket, actions: number[][], now: number) {
    if (this.stopped || !this.pending || ticket.id !== this.pending.id || ticket.index !== this.pending.index || ticket.timestamp !== this.pending.timestamp)
      throw new Error("Stale RTC reply");
    const values = checked(actions), first = Math.max(Math.ceil(this.cursor(now) - EPS), this.applied + 1);
    const base = this.pending.index + this.offset, start = Math.max(first, base), skipped = start - base;
    if (skipped >= 16) throw new Error("RTC reply arrived too late to have a fresh tail");
    for (let i = first; i < base; i++) if (!this.goals.has(i)) {
      if (!this.pending.hold) throw new Error("RTC queue has a gap");
      this.goals.set(i, [...this.pending.hold]);
    }
    for (const index of this.goals.keys()) if (index >= start) this.goals.delete(index);
    for (let i = skipped; i < 16; i++) this.goals.set(base + i, values[i]);
    this.end = base + 16; this.pending = null;
    return { skipped, firstTime: this.anchor + start / 10, remaining: 16 - skipped };
  }
  stop() { this.stopped = true; this.pending = null; this.goals.clear(); }
}
