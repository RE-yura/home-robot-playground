import test from "node:test";
import assert from "node:assert/strict";
import { RtcActionQueue } from "../src/learning/rtc-action-queue.ts";
const chunk = (base = 0) => Array.from({ length: 16 }, (_, i) => [base + i, 0, 0, 0, 0, .5]);
test("RTC preserves the current command and discards the elapsed reply prefix", () => {
  const q = new RtcActionQueue(2); q.prime(chunk());
  assert.equal(q.at(2)[0], 0); const ticket = q.begin(2);
  assert.equal(q.at(2.1)[0], 1); assert.equal(q.at(2.2)[0], 2);
  const accepted = q.accept(ticket, chunk(100), 2.24);
  assert.equal(accepted.skipped, 3);
  assert.equal(q.at(2.25)[0], 2, "no mid-command target replacement");
  assert.equal(q.at(2.3)[0], 103);
  const next = q.begin(2.3); assert.equal(next.prefix[0][0], 103);
  assert.equal(q.at(3.5)[0], 115);
});
test("a reply after latching a boundary cannot replace that already-issued command", () => {
  const q = new RtcActionQueue(0); q.prime(chunk()); q.at(0); const ticket = q.begin(0);
  assert.equal(q.at(.4)[0], 4); const accepted = q.accept(ticket, chunk(100), .4);
  assert.equal(accepted.skipped, 5); assert.equal(q.at(.4)[0], 4); assert.equal(q.at(.5)[0], 105);
});
test("one request at a time; stale and stopped replies cannot replace a queue", () => {
  const q = new RtcActionQueue(0); q.prime(chunk()); const ticket = q.begin(0);
  assert.throws(() => q.begin(0), /pending/i);
  assert.throws(() => q.accept({ ...ticket, id: 999 }, chunk(100), .2), /stale/i);
  assert.equal(q.at(.2)[0], 2); q.stop();
  assert.throws(() => q.accept(ticket, chunk(100), .3), /stale|stopped/i);
});
test("underflow and a reply with no fresh tail are explicit failures", () => {
  const q = new RtcActionQueue(0); q.prime(chunk()); const ticket = q.begin(0);
  assert.throws(() => q.at(1.6), /underflow/i);
  assert.throws(() => q.accept(ticket, chunk(100), 1.6), /fresh|late/i);
});
test("invalid replies cannot partially overwrite the original queue; snapshots do not alias", () => {
  const q = new RtcActionQueue(0), original = chunk(); q.prime(original); original[0][0] = 99;
  assert.equal(q.at(0)[0], 0); const ticket = q.begin(0); ticket.prefix[0][0] = 99;
  assert.equal(q.at(0)[0], 0);
  const bad = chunk(100); bad[15][2] = NaN;
  assert.throws(() => q.accept(ticket, bad, .2), /finite/i);
  assert.equal(q.at(.3)[0], 3);
  assert.throws(() => q.begin(.35), /boundary|pending/i);
});
test("an offset model's reply from capture c fills goals c+2..c+17 and keeps the queued goals before them", () => {
  const q = new RtcActionQueue(0, 2); q.prime(chunk(), [9, 9, 9, 9, 9, .5]);
  assert.equal(q.at(0)[0], 9, "the first offset goals hold the measured state"); assert.equal(q.at(.1)[0], 9);
  assert.equal(q.at(.2)[0], 0); assert.equal(q.at(1.2)[0], 10);
  const ticket = q.begin(1.2); assert.deepEqual(ticket.prefix.map(a => a[0]), [10, 11, 12, 13, 14, 15]);
  const accepted = q.accept(ticket, chunk(100), 1.25);
  assert.equal(accepted.skipped, 0); assert.equal(accepted.firstTime, 1.4);
  assert.equal(q.at(1.3)[0], 11, "goals before c+2 come from the previous chunk"); assert.equal(q.at(1.4)[0], 100); assert.equal(q.at(2.9)[0], 115);
  assert.throws(() => q.at(3), /underflow/i);
});
test("a late offset reply never replaces goals that already ran", () => {
  const q = new RtcActionQueue(0, 2); q.prime(chunk(), [9, 9, 9, 9, 9, .5]);
  q.at(1.2); const ticket = q.begin(1.2); assert.equal(q.at(1.5)[0], 13);
  const accepted = q.accept(ticket, chunk(100), 1.5);
  assert.equal(accepted.skipped, 2); assert.equal(q.at(1.5)[0], 13); assert.equal(q.at(1.6)[0], 102);
});
test("a capture taken with physics held at the queue end holds the last goal until c+2", () => {
  const q = new RtcActionQueue(0, 2); q.prime(chunk(), [9, 9, 9, 9, 9, .5]);
  assert.equal(q.at(1.7)[0], 15); const ticket = q.begin(1.8, true);
  q.accept(ticket, chunk(100), 1.8);
  assert.equal(q.at(1.8)[0], 15); assert.equal(q.at(1.9)[0], 15); assert.equal(q.at(2)[0], 100);
});
test("an offset queue requires a hold goal to prime and rejects unsupported offsets", () => {
  assert.throws(() => new RtcActionQueue(0, 2).prime(chunk()), /hold/i);
  for (const offset of [-1, 1.5, 17]) assert.throws(() => new RtcActionQueue(0, offset), /offset/i);
  const q = new RtcActionQueue(0); q.prime(chunk()); assert.equal(q.at(0)[0], 0);
});
