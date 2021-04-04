import os
from trainml.trainml import TrainML
import asyncio

assert "CLEARML_API_ACCESS_KEY" in os.environ
assert "CLEARML_API_SECRET_KEY" in os.environ


n_gpus = 2
job_cmd = f"""pip install --upgrade pytorch-ignite \
&& pip install fire py-config-runner git+https://github.com/vfdev-5/ImageDatasetViz.git albumentations opencv-python-headless clearml \
&& export CLEARML_API_HOST="https://api.community.clear.ml" \
&& export CLEARML_WEB_HOST="https://app.community.clear.ml" \
&& export CLEARML_FILES_HOST="https://files.community.clear.ml" \
&& CLEARML_API_ACCESS_KEY="{os.environ['CLEARML_API_ACCESS_KEY']}" \
&& CLEARML_API_SECRET_KEY="{os.environ['CLEARML_API_SECRET_KEY']}" \
&& mkdir -p /data/VOCdevkit \
&& ln -s /opt/trainml/input/VOC2012 /data/VOCdevkit/VOC2012 \
&& export DATASET_PATH=/data/ \
&& cd pascal-voc12 \
&& echo "ls" && ls \
&& python -u -m torch.distributed.launch --nproc_per_node={n_gpus} --use_env main.py training configs/baseline_dplv3_resnet101.py
"""


trainml_client = TrainML()

# Create the job
job = asyncio.run(
    trainml_client.jobs.create(
        name="DeeplabV3-PascalVOC12-2GPUs",
        type="headless",
        gpu_type="RTX 2080 Ti",
        gpu_count=2,
        disk_size=10,
        workers=[
            job_cmd,
        ],
        data=dict(
            datasets=[dict(name="PASCAL VOC", type="public")],
        ),
        environment=dict(
            type="DEEPLEARNING_PY38",
        ),
        model=dict(git_uri="https://github.com/vfdev-5/trainml-examples.git"),
    )
)
print(job)

# Watch the log output, attach will return when the training job stops
asyncio.run(job.attach())