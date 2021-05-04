import os
from trainml.trainml import TrainML
import asyncio

assert "CLEARML_API_ACCESS_KEY" in os.environ
assert "CLEARML_API_SECRET_KEY" in os.environ


n_gpus = 4
job_cmd = f"""pip install fire py-config-runner git+https://github.com/vfdev-5/ImageDatasetViz.git albumentations opencv-python-headless clearml \
&& pip install adamp \
&& export CLEARML_API_HOST="https://api.community.clear.ml" \
&& export CLEARML_WEB_HOST="https://app.community.clear.ml" \
&& export CLEARML_FILES_HOST="https://files.community.clear.ml" \
&& export DATASET_PATH=$TRAINML_DATA_PATH/ \
&& export SBD_DATASET_PATH=$TRAINML_DATA_PATH/VOCdevkit/VOCaug/dataset/ \
&& cd pascal-voc12 \
&& nvidia-smi \
&& export config_file=configs/adamp_dplv3_resnet101_sbd.py \
&& python -u -m torch.distributed.launch --nproc_per_node={n_gpus} --use_env main.py training $config_file
"""

trainml_client = TrainML()

# Create the job
job = asyncio.run(
    trainml_client.jobs.create(
        name=f"DeeplabV3-PascalVOC12-{n_gpus}GPUs",
        type="headless",
        gpu_type="RTX 2080 Ti",
        gpu_count=n_gpus,
        disk_size=50,
        workers=[
            job_cmd,
        ],
        data=dict(
            datasets=[dict(name="PASCAL VOC", type="public")],
        ),
        environment=dict(
            type="DEEPLEARNING_PY38",
            env=[
                {"value": f"{os.environ['CLEARML_API_ACCESS_KEY']}", "key": "CLEARML_API_ACCESS_KEY"},
                {"value": f"{os.environ['CLEARML_API_SECRET_KEY']}", "key": "CLEARML_API_SECRET_KEY"},
            ]
        ),
        model=dict(git_uri="https://github.com/vfdev-5/trainml-examples.git"),
    )
)
print(job)

# Watch the log output, attach will return when the training job stops
asyncio.run(job.attach())