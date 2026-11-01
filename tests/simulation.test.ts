import { TABLE_HEIGHT } from "../src/simulation/table.ts";
import assert from "node:assert/strict";
import { test } from "node:test";
import { createRandom } from "../src/simulation/random.ts";
import { createObjects } from "../src/simulation/scene.ts";

test("seed preserves table and floor objects in separate safe slots outside the bin", () => {
  assert.deepEqual(createObjects(42), createObjects(42));
  assert.notDeepEqual(createObjects(42), createObjects(43));
  for (let seed = 0; seed < 100; seed++) {
    const objects = createObjects(seed);
    assert.equal(new Set(objects.map(o => o.type)).size, objects.length, "each object kind appears only once for every seed");
    assert.deepEqual(objects.map(o => o.type), ["cube", "ball", "can", "mug", "bottle", "plate", "baseball", "tea", "juice", "milk"]);
    assert.equal(new Set(objects.map(o => o.id)).size, 10);
    for (const object of objects) {
      if (object.id === "baseball") {
        assert.ok(object.position[1] > 0.2 && object.position[1] < 0.4);
        assert.ok(Math.abs(object.position[0]) > 0.8 || Math.abs(object.position[2]) > 0.5, "floor slot is outside the table");
      } else if (["tea", "juice", "milk"].includes(object.id)) {
        assert.deepEqual(object.position, createObjects(42).find(o => o.id === object.id)!.position, "fridge shelf positions stay safe for every seed");
      } else {
        assert.ok(object.position[1] > TABLE_HEIGHT + 0.15);
        assert.ok(object.position[0] < 0.24);
      }
      for (const other of objects) {
        if (object.id === other.id) continue;
        assert.ok(Math.hypot(object.position[0] - other.position[0], object.position[2] - other.position[2]) > 0.2);
      }
    }
  }
});

test("seeded randomness is stable and bounded", () => {
  const random = createRandom(42);
  assert.equal(random(), 0.6011037519201636);
  for (let index = 0; index < 10000; index++) assert.ok(random() < 1);
  assert.equal(createRandom(-1)(), createRandom(4294967295)());
});
