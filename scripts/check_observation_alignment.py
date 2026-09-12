#!/usr/bin/env python3
import io, json, sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from pi05_local import environment as env

DATA=Path('/root/shared-nvme/pi05_panda_multi_object_box_v2/dataset/train')
PARQUET=DATA/'data/chunk-000/file-000.parquet'
META=DATA/'meta/pi05_episodes.jsonl'
rows=[json.loads(x) for x in META.read_text().splitlines()]
ep=rows[0]
t=pq.read_table(PARQUET).slice(0,1).to_pydict()
def decode(k): return np.asarray(Image.open(io.BytesIO(t[k][0]['bytes'])).convert('RGB'))
train_ext,train_wrist=decode('observation.images.image'),decode('observation.images.wrist_image')
spec=env.TaskSpec(seed=ep['scene_seed'],positions=ep['positions'],tray_xy=ep['tray_xy'],target_type=ep['target_type'],recovery_type=ep['recovery_type'],split=ep['split'])
model,data,ids=env.build_task_model(spec)
renderer=env.RenderPair(model,size=256)
render_ext,render_wrist=renderer.render(data)
state=np.concatenate([data.qpos[ids.arm_qpos],[env.normalized_gripper_state(data,ids)]])

def stats(a): return {'shape':list(a.shape),'dtype':str(a.dtype),'min':float(a.min()),'max':float(a.max()),'mean':float(a.mean()),'std':float(a.std())}
def compare(a,b):
    d=np.abs(a.astype(np.float32)-b.astype(np.float32))
    return {'mae':float(d.mean()),'max_abs':float(d.max()),'same_shape':a.shape==b.shape}
report={'episode_meta':{k:ep[k] for k in ('episode_index','scene_seed','positions','tray_xy','target_type','recovery_type')},'train_state':np.asarray(t['observation.state'][0]).tolist(),'render_state':state.tolist(),'state_abs_err':float(np.abs(np.asarray(t['observation.state'][0])-state).max()),'train_timestamp':float(t['timestamp'][0]),'train_frame_index':int(t['frame_index'][0]),'image_train':stats(train_ext),'image_render':stats(render_ext),'image_compare':compare(train_ext,render_ext),'wrist_train':stats(train_wrist),'wrist_render':stats(render_wrist),'wrist_compare':compare(train_wrist,render_wrist)}
Path('/root/shared-nvme/logs/observation_alignment.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
renderer.close()
