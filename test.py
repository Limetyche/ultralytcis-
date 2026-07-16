import torch
from ultralytics import YOLO

model = YOLO("/content/drive/MyDrive/ultralytics/ultralytics/cfg/models/13/yolo13.yaml").model.cuda().train()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
)

scaler = torch.amp.GradScaler("cuda")

for step in range(100):
    x = torch.randn(
        2, 3, 640, 640,
        device="cuda",
    )

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(
        "cuda",
        dtype=torch.float16,
    ):
        outputs = model(x)

        # 这里只做结构压力测试，不是实际检测 loss
        tensors = []

        def collect(obj):
            if torch.is_tensor(obj):
                tensors.append(obj)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    collect(item)
            elif isinstance(obj, dict):
                for item in obj.values():
                    collect(item)

        collect(outputs)

        loss = sum(
            t.float().square().mean()
            for t in tensors
        )

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Non-finite loss at step {step}"
        )

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    for name, p in model.named_parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                raise RuntimeError(
                    f"Bad grad at step {step}: {name}"
                )

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        10.0,
    )

    scaler.step(optimizer)
    scaler.update()

    if step % 10 == 0:
        print(
            step,
            loss.detach().item(),
            scaler.get_scale(),
        )