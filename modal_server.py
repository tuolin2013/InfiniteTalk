"""
InfiniteTalk Modal 部署服务（音频驱动数字人口播视频 — 无限时长）

能力：
  1. 图片 + 音频 → 对口型说话视频（Image-to-Video，单人）
  2. 视频 + 音频 → 换音说话视频（Video-to-Video，稀疏帧配音，单人）
  3. 多人版（multi）同上，InfiniteTalk multi 权重

架构说明：
- InfiniteTalk 基于 Wan2.1-I2V-14B-480P + 自研 AudioCondition LoRA
- 权重下载到 Modal Volume /weights/
- 入参：multipart/form-data（图片/视频 + 音频）
- 出参：R2 公网 MP4 URL（避免把大文件回传国内后端）

部署命令：
    cd e:/laotuo_project/InfiniteTalk
    modal deploy modal_server.py

配套 Secrets（必须提前创建，只需 r2-storage）：
    modal secret create r2-storage \\
        R2_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com \\
        R2_ACCESS_KEY_ID=xxx \\
        R2_SECRET_ACCESS_KEY=xxx \\
        R2_BUCKET_NAME=ai-products \\
        R2_PUBLIC_DOMAIN=https://assets.laotuo.top

鉴权说明：
  不使用 Modal Secret 做 token 鉴权，直接依赖后端代理（OmniSKU-Forge backend）
  在调用前验证用户身份，Modal 端点对后端 IP 开放即可。
"""

import io
import os
import modal

# ─────────────────────────────────────────────
# 1. 核心资源
# ─────────────────────────────────────────────
app = modal.App("infinitetalk-api-factory")

# 持久化 Volume：源码 + 预训练权重
volume = modal.Volume.from_name("infinitetalk-weights-vault", create_if_missing=True)

REPO_DIR = "/weights/InfiniteTalk"
WEIGHTS_DIR = "/weights"

# R2 Secret（与 SadTalker 同一个 secret）
r2_secret = modal.Secret.from_name(
    "r2-storage",
    required_keys=[
        "R2_ENDPOINT_URL",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET_NAME",
        "R2_PUBLIC_DOMAIN",
    ],
)

# HuggingFace token（避免匿名下载限速 / 需要鉴权的模型）
# 提前创建: modal secret create hf-token HF_TOKEN=hf_xxxx
hf_secret = modal.Secret.from_name("hf-token", required_keys=["HF_TOKEN"])


def _upload_to_r2(video_bytes: bytes, ext: str = "mp4", content_type: str = "video/mp4") -> str:
    """在 Modal 容器内把视频字节上传到 R2，返回公网可访问 URL。"""
    import uuid
    import boto3
    from botocore.config import Config as BotoConfig

    endpoint = os.environ["R2_ENDPOINT_URL"]
    bucket = os.environ["R2_BUCKET_NAME"]
    public_domain = os.environ["R2_PUBLIC_DOMAIN"].rstrip("/")
    if not public_domain.startswith(("http://", "https://")):
        public_domain = f"https://{public_domain}"

    file_key = f"uploads/infinitetalk_{uuid.uuid4().hex}.{ext}"
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("R2_REGION", "auto"),
        config=BotoConfig(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "standard"},
            connect_timeout=30,
            read_timeout=300,
        ),
    )
    print(f"☁️ 上传到 R2: bucket={bucket} key={file_key} ({len(video_bytes)//1024}KB)")
    s3.put_object(Bucket=bucket, Key=file_key, Body=video_bytes, ContentType=content_type)
    url = f"{public_domain}/{file_key}"
    print(f"✅ R2 上传完成: {url}")
    return url


# ─────────────────────────────────────────────
# 2. 容器镜像
# ─────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "ffmpeg",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libsndfile1",
        "wget",
        "curl",
    )
    # 先装 packaging，flash_attn setup.py 依赖它
    .pip_install("packaging", "ninja")
    # torch 2.4.1 + torchvision 0.19.1（CUDA 12.1）— InfiniteTalk 官方推荐
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "torchaudio==2.4.1",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "xformers==0.0.28",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    # flash_attn 预编译轮子：直接用 GitHub Release 的 cp310-cu121-torch2.4 二进制包
    # 完全绕过本地编译，无需 nvcc / CUDA_HOME
    .run_commands(
        "pip install --no-build-isolation "
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
        "flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    )
    .pip_install(
        # InfiniteTalk requirements.txt
        "opencv-python>=4.9.0.80",
        # diffusers 0.33.0 引入了 attention_dispatch.py，其中用到的
        # flash_attn_3 custom op 注册方式需要 torch 2.5+，与此处 torch 2.4.1
        # 不兼容（ValueError: infer_schema unsupported type torch.Tensor）。
        # 固定在 0.32.2（最后一个兼容 torch 2.4 的版本）。
        "diffusers==0.32.2",
        "transformers>=4.49.0",
        "tokenizers>=0.20.3",
        "accelerate>=1.1.1",
        "tqdm",
        "imageio",
        "easydict",
        "ftfy",
        "imageio-ffmpeg",
        "scikit-image",
        "loguru",
        "numpy>=1.23.5,<2",
        "xfuser>=0.4.1",
        "pyloudnorm",
        "optimum-quanto==0.2.6",
        "scenedetect",
        "moviepy==1.0.3",
        "decord",
        # 语音处理
        "librosa",
        "soundfile",
        "misaki[en]",
        # 部署相关
        "fastapi[standard]",
        "python-multipart",
        "requests",
        "boto3",
        "huggingface_hub",
        "hf_transfer",
        "Pillow",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONUNBUFFERED": "1",
        # HF_TOKEN 在运行时由 hf_secret 注入，这里设置 hub 行为
        "HF_HUB_DISABLE_PROGRESS_BARS": "0",
    })
)


# ─────────────────────────────────────────────
# 3. 服务主体（GPU 工作类）
# ─────────────────────────────────────────────
@app.cls(
    image=image,
    gpu="a100",          # InfiniteTalk 14B 模型至少需要 40GB VRAM，选 A100-40GB
    volumes={WEIGHTS_DIR: volume},
    secrets=[r2_secret, hf_secret],
    scaledown_window=300,
    startup_timeout=1800,   # 首次下载 14B 权重约需 15~20 分钟
    timeout=3600,           # 长视频生成可能需要 30~60 分钟
)
@modal.concurrent(max_inputs=1)  # 单 A100 一次只跑一个任务，避免 OOM
class InfiniteTalkService:

    @modal.enter()
    def setup(self):
        """冷启动：克隆源码 + 下载模型权重到 Volume"""
        import subprocess, sys

        # ── Step 1: 克隆 / 更新 InfiniteTalk 源码 ────────────────
        if not os.path.exists(os.path.join(REPO_DIR, "generate_infinitetalk.py")):
            print("📦 首次运行：克隆 InfiniteTalk 源码到 Volume...")
            subprocess.run(
                ["git", "clone", "https://github.com/tuolin2013/InfiniteTalk.git", REPO_DIR],
                check=True,
            )
            volume.commit()
            print("✅ InfiniteTalk 源码克隆完成")
        else:
            print(f"✅ InfiniteTalk 源码已存在: {REPO_DIR}，执行 git pull 更新...")
            result = subprocess.run(
                ["git", "pull", "--ff-only"],
                cwd=REPO_DIR,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"✅ git pull 完成: {result.stdout.strip()}")
                volume.commit()
            else:
                print(f"⚠️ git pull 失败（继续使用现有版本）: {result.stderr.strip()}")

        sys.path.insert(0, REPO_DIR)

        # ── Step 2: 下载模型权重 ─────────────────────────────────
        from huggingface_hub import snapshot_download, hf_hub_download, login as hf_login

        # 用 HF_TOKEN 登录，避免匿名下载限速 / 需要鉴权的模型
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            hf_login(token=hf_token, add_to_git_credential=False)
            print("✅ HuggingFace 已登录（有 token）")
        else:
            print("⚠️ 未设置 HF_TOKEN，以匿名模式下载（可能被限速）")

        wan_dir = os.path.join(WEIGHTS_DIR, "Wan2.1-I2V-14B-480P")
        wav2vec_dir = os.path.join(WEIGHTS_DIR, "chinese-wav2vec2-base")
        infinitetalk_dir = os.path.join(WEIGHTS_DIR, "InfiniteTalk")

        # 2a. Wan2.1-I2V-14B-480P 基础模型（~28GB）
        if not os.path.exists(os.path.join(wan_dir, "config.json")):
            print("🚚 下载 Wan2.1-I2V-14B-480P 基础模型（~28GB，首次较慢）...")
            snapshot_download(
                repo_id="Wan-AI/Wan2.1-I2V-14B-480P",
                local_dir=wan_dir,
                token=hf_token or None,
            )
            volume.commit()
            print("✅ Wan2.1-I2V-14B-480P 下载完成")
        else:
            print("✅ Wan2.1-I2V-14B-480P 已存在")

        # 2b. chinese-wav2vec2-base 音频编码器
        if not os.path.exists(os.path.join(wav2vec_dir, "config.json")):
            print("🚚 下载 chinese-wav2vec2-base...")
            snapshot_download(
                repo_id="TencentGameMate/chinese-wav2vec2-base",
                local_dir=wav2vec_dir,
                token=hf_token or None,
            )
            # 额外下载 safetensors 版本权重（refs/pr/1）
            try:
                hf_hub_download(
                    repo_id="TencentGameMate/chinese-wav2vec2-base",
                    filename="model.safetensors",
                    revision="refs/pr/1",
                    local_dir=wav2vec_dir,
                    token=hf_token or None,
                )
            except Exception as e:
                print(f"   ⚠️ safetensors 版本下载失败（可忽略）: {e}")
            volume.commit()
            print("✅ chinese-wav2vec2-base 下载完成")
        else:
            print("✅ chinese-wav2vec2-base 已存在")

        # 2c. InfiniteTalk LoRA 权重（单人 + 多人）
        it_single = os.path.join(infinitetalk_dir, "single", "infinitetalk.safetensors")
        if not os.path.exists(it_single):
            print("🚚 下载 MeiGen-AI/InfiniteTalk 权重...")
            snapshot_download(
                repo_id="MeiGen-AI/InfiniteTalk",
                local_dir=infinitetalk_dir,
                token=hf_token or None,
            )
            volume.commit()
            print("✅ InfiniteTalk 权重下载完成")
        else:
            print("✅ InfiniteTalk 权重已存在")

        print("✅ InfiniteTalk 服务就绪！")

    def _run_infinitetalk(
        self,
        source_bytes: bytes,
        audio_bytes: bytes,
        source_is_video: bool = False,
        mode: str = "streaming",
        size: str = "infinitetalk-480",
        sample_steps: int = 40,
        motion_frame: int = 9,
        multi: bool = False,
        num_persistent_param_in_dit: int = 0,
        max_frame_num: int = 1000,
    ) -> bytes:
        """调用 InfiniteTalk generate_infinitetalk.py（子进程），返回 MP4 字节。"""
        import tempfile, subprocess, glob, shutil, json

        work = tempfile.mkdtemp(dir="/tmp")
        aud_path = os.path.join(work, "audio.wav")
        out_path = os.path.join(work, "infinitetalk_result.mp4")

        # 落盘音频：统一转 16k 单声道 wav
        raw_aud = os.path.join(work, "audio_raw")
        with open(raw_aud, "wb") as f:
            f.write(audio_bytes)
        subprocess.run(
            ["ffmpeg", "-y", "-i", raw_aud, "-ar", "16000", "-ac", "1", aud_path],
            check=True, capture_output=True,
        )

        # 落盘源文件（图片或视频）
        if source_is_video:
            src_path = os.path.join(work, "source.mp4")
            with open(src_path, "wb") as f:
                f.write(source_bytes)
        else:
            src_path = os.path.join(work, "source.png")
            from PIL import Image as _PIL
            _PIL.open(io.BytesIO(source_bytes)).convert("RGB").save(src_path)

        # 构建 input_json（generate_infinitetalk.py 期望的格式）
        # 脚本读取: input_data['cond_video'], input_data['cond_audio']['person1'],
        #           input_data['prompt']（可选）, input_data['audio_type']（可选）
        input_data = {
            "cond_video": src_path,
            "cond_audio": {"person1": aud_path},
            "prompt": "",
        }
        input_json_path = os.path.join(work, "input.json")
        with open(input_json_path, "w") as f:
            json.dump(input_data, f)

        # audio_save_dir：脚本用 cond_video 文件名派生子目录，需提前存在
        audio_save_dir = os.path.join(work, "audio_emb")
        os.makedirs(audio_save_dir, exist_ok=True)

        # 选择权重路径
        if multi:
            infinitetalk_weights = os.path.join(WEIGHTS_DIR, "InfiniteTalk", "multi", "infinitetalk.safetensors")
        else:
            infinitetalk_weights = os.path.join(WEIGHTS_DIR, "InfiniteTalk", "single", "infinitetalk.safetensors")

        cmd = [
            "python", "generate_infinitetalk.py",
            "--ckpt_dir", os.path.join(WEIGHTS_DIR, "Wan2.1-I2V-14B-480P"),
            "--wav2vec_dir", os.path.join(WEIGHTS_DIR, "chinese-wav2vec2-base"),
            "--infinitetalk_dir", infinitetalk_weights,
            "--input_json", input_json_path,
            "--audio_save_dir", audio_save_dir,
            "--size", size,
            "--sample_steps", str(sample_steps),
            "--mode", mode,
            "--motion_frame", str(motion_frame),
            "--num_persistent_param_in_dit", str(num_persistent_param_in_dit),
            "--max_frame_num", str(max_frame_num),
            "--save_file", os.path.join(work, "infinitetalk_result"),
        ]

        print(f"🎬 运行 InfiniteTalk: {' '.join(cmd)}", flush=True)

        import collections, time
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(
            cmd, cwd=REPO_DIR, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        tail = collections.deque(maxlen=80)
        t0 = time.time()
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                tail.append(line)
                print(f"   [InfiniteTalk] {line}", flush=True)
        proc.wait()
        elapsed = int(time.time() - t0)
        print(f"⏱️ InfiniteTalk 子进程结束 returncode={proc.returncode}, 耗时 {elapsed}s", flush=True)

        if proc.returncode != 0:
            raise RuntimeError(
                "InfiniteTalk 推理失败 (returncode={})\n最后日志:\n{}".format(
                    proc.returncode, "\n".join(tail)
                )
            )

        # 查找输出视频（generate_infinitetalk.py 会在 save_file 前缀旁生成 .mp4）
        mp4s = sorted(
            glob.glob(os.path.join(work, "**", "*.mp4"), recursive=True),
            key=os.path.getmtime, reverse=True,
        )
        if not mp4s:
            raise RuntimeError(f"InfiniteTalk 完成但未找到输出视频，目录内容: {os.listdir(work)}")

        with open(mp4s[0], "rb") as f:
            data = f.read()
        shutil.rmtree(work, ignore_errors=True)
        print(f"✅ InfiniteTalk 输出 {len(data)//1024}KB")
        return data

    @modal.method()
    def talk_i2v(
        self,
        image_bytes: bytes,
        audio_bytes: bytes,
        mode: str = "streaming",
        size: str = "infinitetalk-480",
        sample_steps: int = 40,
        motion_frame: int = 9,
        multi: bool = False,
        max_frame_num: int = 1000,
    ) -> str:
        """图片 + 音频 → 数字人口播视频，上传 R2 返回公网 URL。"""
        if not image_bytes:
            raise ValueError("源图片为空")
        if not audio_bytes:
            raise ValueError("音频为空")
        video = self._run_infinitetalk(
            image_bytes, audio_bytes,
            source_is_video=False,
            mode=mode, size=size,
            sample_steps=sample_steps,
            motion_frame=motion_frame,
            multi=multi,
            max_frame_num=max_frame_num,
        )
        return _upload_to_r2(video)

    @modal.method()
    def talk_v2v(
        self,
        video_bytes: bytes,
        audio_bytes: bytes,
        mode: str = "streaming",
        size: str = "infinitetalk-480",
        sample_steps: int = 40,
        motion_frame: int = 9,
        max_frame_num: int = 1000,
    ) -> str:
        """视频 + 音频 → 替换音轨并重新同步口型（V2V 配音模式），上传 R2 返回 URL。"""
        if not video_bytes:
            raise ValueError("源视频为空")
        if not audio_bytes:
            raise ValueError("音频为空")
        video = self._run_infinitetalk(
            video_bytes, audio_bytes,
            source_is_video=True,
            mode=mode, size=size,
            sample_steps=sample_steps,
            motion_frame=motion_frame,
            multi=False,
            max_frame_num=max_frame_num,
        )
        return _upload_to_r2(video)

    # ─────────────────────────────────────────────
    # 4. HTTP 端点（FastAPI ASGI — 挂载在 GPU 类上）
    #
    # 关键修复：fastapi_app 必须是 @app.cls 上的 @modal.asgi_app() 方法，
    # 而不是独立的 @app.function。独立函数没有 GPU / Volume / Secrets，
    # 导致 InfiniteTalkService() 实例化后无法 spawn 任何任务，也读不到
    # 权重和 R2 凭据——表现就是「没有动静」。
    # ─────────────────────────────────────────────
    @modal.asgi_app()
    def fastapi_app(self):
        from fastapi import FastAPI, UploadFile, Form
        from fastapi.responses import JSONResponse
        from fastapi.middleware.cors import CORSMiddleware
        from typing import Annotated

        web_app = FastAPI(title="InfiniteTalk API")
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["Content-Disposition"],
        )

        # 持有对当前 cls 实例的引用，供路由闭包使用
        svc = self

        @web_app.post("/talk")
        async def talk(
            source_image: UploadFile,
            driven_audio: UploadFile,
            mode: Annotated[str, Form()] = "streaming",
            size: Annotated[str, Form()] = "infinitetalk-480",
            sample_steps: Annotated[int, Form()] = 40,
            motion_frame: Annotated[int, Form()] = 9,
            multi: Annotated[bool, Form()] = False,
            max_frame_num: Annotated[int, Form()] = 1000,
        ):
            """
            数字人口播：图片 + 音频 → 对口型说话视频（I2V，异步 spawn 模式）

            请求：multipart/form-data
              - source_image : 人物正脸图（JPG/PNG）
              - driven_audio : 驱动音频（wav/mp3/m4a 等）
              - mode         : streaming（无限时长）或 clip（单段短视频）
              - size         : infinitetalk-480 或 infinitetalk-720
              - sample_steps : 采样步数，默认 40（越高质量越好但越慢）
              - motion_frame : 运动帧参数，默认 9
              - multi        : 是否使用多人版权重（默认 false）
              - max_frame_num: 最大帧数，1000 ≈ 40秒

            返回：{"call_id": "..."}，轮询 GET /result/{call_id} 获取结果。
            """
            img = await source_image.read()
            aud = await driven_audio.read()
            print(f"📥 /talk | image={source_image.filename!r}({len(img)}B) audio={driven_audio.filename!r}({len(aud)}B)")

            call = svc.talk_i2v.spawn(
                image_bytes=img,
                audio_bytes=aud,
                mode=mode,
                size=size,
                sample_steps=sample_steps,
                motion_frame=motion_frame,
                multi=multi,
                max_frame_num=max_frame_num,
            )
            return JSONResponse(content={"call_id": call.object_id})

        @web_app.post("/talk-v2v")
        async def talk_v2v(
            source_video: UploadFile,
            driven_audio: UploadFile,
            mode: Annotated[str, Form()] = "streaming",
            size: Annotated[str, Form()] = "infinitetalk-480",
            sample_steps: Annotated[int, Form()] = 40,
            motion_frame: Annotated[int, Form()] = 9,
            max_frame_num: Annotated[int, Form()] = 1000,
        ):
            """
            视频配音：源视频 + 新音频 → 重新同步口型的说话视频（V2V，异步 spawn 模式）
            """
            vid = await source_video.read()
            aud = await driven_audio.read()
            print(f"📥 /talk-v2v | video={source_video.filename!r}({len(vid)}B) audio={len(aud)}B")

            call = svc.talk_v2v.spawn(
                video_bytes=vid,
                audio_bytes=aud,
                mode=mode,
                size=size,
                sample_steps=sample_steps,
                motion_frame=motion_frame,
                max_frame_num=max_frame_num,
            )
            return JSONResponse(content={"call_id": call.object_id})

        @web_app.get("/result/{call_id}")
        async def result(call_id: str):
            """
            轮询任务结果。

            - 仍在渲染：HTTP 202 + {"status": "pending"}
            - 已完成  ：HTTP 200 + {"status": "done", "url": "<R2 公网地址>"}
            - 失败    ：HTTP 500 + {"status": "error", "message": "..."}
            """
            from modal.functions import FunctionCall

            try:
                fc = FunctionCall.from_id(call_id)
            except Exception as exc:
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail=f"无效的 call_id: {exc}")

            try:
                url = await fc.get.aio(timeout=0)
            except TimeoutError:
                return JSONResponse(status_code=202, content={"status": "pending"})
            except Exception as exc:
                return JSONResponse(
                    status_code=500,
                    content={"status": "error", "message": str(exc)[:800]},
                )

            return JSONResponse(content={"status": "done", "url": url})

        @web_app.get("/health")
        async def health():
            """服务健康检查。"""
            return {"status": "ok", "service": "InfiniteTalk"}

        return web_app
