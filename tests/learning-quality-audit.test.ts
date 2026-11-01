import assert from 'node:assert/strict';
import {test} from 'node:test';
import {unintendedContactForce} from '../src/learning/quality.ts';

test('closed fingers on the basket or the other object remain unintended contact',()=>{
 const contact={robotBodies:1,finger:true,floor:false,objectId:'basket',normalForce:120,grasping:true};
 assert.equal(unintendedContactForce([contact],'mug'),120);
 assert.equal(unintendedContactForce([{...contact,objectId:'ball'}],'mug'),120);
 assert.equal(unintendedContactForce([{...contact,objectId:'mug'}],'mug'),0);
 assert.equal(unintendedContactForce([{...contact,finger:false,objectId:'mug'}],'mug'),120);
 assert.equal(unintendedContactForce([{...contact,objectId:'mug',grasping:false}],'mug'),120,'open-finger impact is not an established grasp');
});
