import os.path as osp
import time
from mmseg.online_training.api_server_sly import create_api
from mmseg.online_training.request_queue import RequestQueue, RequestType
from mmseg.engine.hooks.online_training_api_sly import OnlineTrainingAPISly
import uvicorn
import threading
import asyncio
import torch
import supervisely as sly
from mmengine.runner.checkpoint import CheckpointLoader
from mmengine.config import Config
from mmengine.runner import Runner


BASE_CONFIG_FILE = "configs/mask2former/custom/swin-l-online-app.py"
RESUME_FROM_CHECKPOINT = None
API_HOST = "0.0.0.0"
API_PORT = 8000
INITIAL_SAMPLES = 2
DEVELOP_AND_DEBUG = True


@CheckpointLoader.register_scheme(prefixes='', force=True)
def load_from_local(filename, map_location):
    filename = osp.expanduser(filename)
    if not osp.isfile(filename):
        raise FileNotFoundError(f'{filename} can not be found.')
    checkpoint = torch.load(filename, map_location=map_location, weights_only=False)
    return checkpoint


def start_api_server(
    request_queue: RequestQueue,
    host: str = "0.0.0.0",
    port: int = 8000
) -> threading.Thread:
    app = sly.Application()
    server = app.get_server()
    create_api(server, request_queue)

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    
    thread = threading.Thread(target=server.run, daemon=True, name="APIServer")
    thread.start()
    
    print(f"🚀 Online Training API Server Started")
    print(f"   URL: http://{host}:{port}")
    print(f"   POST /start        - Start online training")
    print(f"   POST /predict      - Run inference")
    print(f"   POST /add-sample   - Add training sample")
    print(f"   POST /status       - Check training status")

    return thread


def wait_for_start(request_queue: RequestQueue):
    print("⏳ Waiting for /start request...")
    
    while True:
        requests = request_queue.get_all()
        
        for request_type, data, future in requests:
            if request_type == RequestType.START:
                result = {"status": "success", "message": "Online training will start now."}
                threading.Thread(
                    target=lambda: asyncio.run(_set_future_result(future, result))
                ).start()
                return
            else:
                request_queue._queue.put((request_type, data, future))
        
        time.sleep(0.5)


async def _set_future_result(future: asyncio.Future, result):
    if not future.done():
        future.set_result(result)


def prepare_training(api: sly.Api, project_id: int):
    meta_json = api.project.get_meta(project_id)
    project_meta = sly.ProjectMeta.from_json(meta_json)
    
    obj_classes_filtered = [
        obj_class for obj_class in project_meta.obj_classes
        if obj_class.geometry_type is sly.Bitmap
    ]
    classes = tuple([obj_class.name for obj_class in obj_classes_filtered])
    palette = [tuple(obj_class.color) for obj_class in obj_classes_filtered]
    num_classes = len(classes)
    metainfo = dict(classes=classes, palette=palette)
    print(f"📋 Detected {num_classes} classes: {classes}")
    assert num_classes > 0, "No bitmap object classes found in project!"
    class_collection = sly.ObjClassCollection(obj_classes_filtered)
    
    cfg = Config.fromfile(BASE_CONFIG_FILE)

    data_root = f'app_data/sly_project_{project_id}/'
    work_dir = f'app_data/sly_project_{project_id}/experiments/'

    cfg.work_dir = work_dir
    cfg.classes = classes
    cfg.num_classes = num_classes
    cfg.metainfo = metainfo
    cfg.data_root = data_root
    assert cfg.test_pipeline is not None, "test_pipeline is not defined in the config!"

    if RESUME_FROM_CHECKPOINT is not None:
        cfg.resume = True
        cfg.load_from = RESUME_FROM_CHECKPOINT
        print(f"🔄 Resuming training from checkpoint: {RESUME_FROM_CHECKPOINT}")
    
    cfg.model.decode_head.num_classes = num_classes
    cfg.model.decode_head.loss_cls.class_weight = [1.0] * num_classes + [0.1]
    
    cfg.train_dataloader.dataset.metainfo = metainfo
    cfg.train_dataloader.dataset.data_root = data_root

    return cfg, class_collection, data_root
    

if __name__ == "__main__":
    if DEVELOP_AND_DEBUG and not sly.is_production():
        print("🔧 Initializing Develop & Debug application...")
        team_id = sly.env.team_id()
        sly.app.development.supervisely_vpn_network(action="up")
        task = sly.app.development.create_debug_task(team_id, port="8000")

    api = sly.Api()
    project_id = sly.env.project_id()

    request_queue = RequestQueue()
    api_thread = start_api_server(request_queue, host=API_HOST, port=API_PORT)

    print("🏁 Starting online training...")
    cfg, obj_classes_filtered, data_root = prepare_training(api, project_id)
    
    print("⚙️ Initializing MMEngine Runner...")
    runner = Runner.from_cfg(cfg)
    
    custom_hook = OnlineTrainingAPISly(
        request_queue=request_queue,
        obj_classes=obj_classes_filtered,
        images_dir=data_root + 'images/train',
        start_samples=INITIAL_SAMPLES,
        warmup_iters=60,
        score_thr=0.3,
    )
    runner.register_custom_hooks([custom_hook])

    print("🚀 Starting training loop...")
    runner.train()