

    

eval_single_cfg = {
    "model_path": "/home/beihang/jzl/models/zai-org/CogVideoX-5b",
    "hidden_size": 4096,
    "rank": 256,
    "lr": 5e-4,
    "device": "cuda",
    "adapter_path": "checkpoints/4/safe_adapter.pt",
    "cls_ckpt_path": "checkpoints/prompt_classifier.pt",
    "testset_path": "datasets/train/4.csv",
    "output_dir": "out/4_cls",
    "num_frames": 81,
    "height": 480,
    "width": 720,
    "num_inference_steps": 50,
    "guidance_scale": 6.0,
    "use_benign": False,
    "save_every": 5,
    "fps": 16,
}

eval_multi_cfg = {
    "model_path": "/home/beihang/jzl/models/zai-org/CogVideoX-5b",
    "hidden_size": 4096,
    "rank": 256,
    "lr": 5e-4,
    "device": "cuda",
    "adapter_ckpt_map": {
        "sexual": "checkpoints/sexual_adapter.pt",
        "violent": "checkpoints/violent_adapter.pt",
        "political": "checkpoints/political_adapter.pt",
        "disturbing": "checkpoints/disturbing_adapter.pt",
    },
    "route_thresh": 0.3,
    "cls_ckpt_path": "checkpoints/prompt_classifier.pt",
    "testset_path": "datasets/train/4.csv",
    "output_dir": "out/4_multi_cls",
    "num_frames": 81,
    "height": 480,
    "width": 720,
    "num_inference_steps": 50,
    "guidance_scale": 6.0,
    "use_benign": False,
    "save_every": 5,
    "fps": 16,
}