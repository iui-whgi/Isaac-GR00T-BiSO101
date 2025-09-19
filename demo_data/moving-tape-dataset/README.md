---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
configs:
- config_name: default
  data_files: data/*/*.parquet
---

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

## Dataset Description



- **Homepage:** [More Information Needed]
- **Paper:** [More Information Needed]
- **License:** apache-2.0

## Dataset Structure

[meta/info.json](meta/info.json):
```json
{
    "codebase_version": "v2.1",
    "robot_type": "dual_so101",
    "total_episodes": 15,
    "total_frames": 5418,
    "total_tasks": 1,
    "total_videos": 45,
    "total_chunks": 1,
    "chunks_size": 1000,
    "fps": 30,
    "splits": {
        "train": "0:15"
    },
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "features": {
        "action": {
            "dtype": "float32",
            "shape": [
                12
            ],
            "names": [
                "shoulder_pan_left.pos",
                "shoulder_lift_left.pos",
                "elbow_flex_left.pos",
                "wrist_flex_left.pos",
                "wrist_roll_left.pos",
                "gripper_left.pos",
                "shoulder_pan_right.pos",
                "shoulder_lift_right.pos",
                "elbow_flex_right.pos",
                "wrist_flex_right.pos",
                "wrist_roll_right.pos",
                "gripper_right.pos"
            ]
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [
                12
            ],
            "names": [
                "shoulder_pan_left.pos",
                "shoulder_lift_left.pos",
                "elbow_flex_left.pos",
                "wrist_flex_left.pos",
                "wrist_roll_left.pos",
                "gripper_left.pos",
                "shoulder_pan_right.pos",
                "shoulder_lift_right.pos",
                "elbow_flex_right.pos",
                "wrist_flex_right.pos",
                "wrist_roll_right.pos",
                "gripper_right.pos"
            ]
        },
        "observation.images.cam0": {
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "video.fps": 30,
                "video.channels": 3,
                "has_audio": false
            }
        },
        "observation.images.cam1": {
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "video.fps": 30,
                "video.channels": 3,
                "has_audio": false
            }
        },
        "observation.images.cam2": {
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "video.fps": 30,
                "video.channels": 3,
                "has_audio": false
            }
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": null
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "task_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        }
    }
}
```


## Citation

**BibTeX:**

```bibtex
[More Information Needed]
```