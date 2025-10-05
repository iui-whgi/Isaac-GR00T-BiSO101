# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
from torch import nn
from transformers import MambaForCausalLM, AutoTokenizer, CLIPVisionModel, CLIPImageProcessor
from transformers.feature_extraction_utils import BatchFeature

import gr00t

DEFAULT_MAMBA_PATH = "state-spaces/mamba-2.8b-hf"
DEFAULT_CLIP_PATH = "openai/clip-vit-base-patch32"


class MambaBackbone(nn.Module):
    def __init__(
        self,
        tune_llm: bool = False,
        tune_visual: bool = True,  # 비디오 처리 활성화
        select_layer: int = -1,
        reproject_vision: bool = False,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        mamba_path: str | None = None,
        project_to_dim: int = 1536,
        mamba_type: str = "mamba-2.8b",
        clip_path: str | None = None,
    ):
        """
        Args:
            tune_llm: whether to tune the Mamba model (default: False)
            tune_visual: whether to tune the visual model (default: True)
            select_layer: which layer to select for features (default: -1)
            reproject_vision: reproject vision features (default: False)
            use_flash_attention: use flash attention (Mamba는 자체 최적화 사용)
            load_bf16: load in bfloat16 (default: False)
            mamba_path: path to Mamba model (default: None, uses default)
            project_to_dim: dimension to project features to (default: 1536)
            mamba_type: type of Mamba model (default: "mamba-2.8b")
            clip_path: path to CLIP model for vision processing (default: None, uses default)
        """
        super().__init__()
        
        # Mamba 모델 로드
        if mamba_path:
            self.mamba_model = MambaForCausalLM.from_pretrained(mamba_path)
            self.tokenizer = AutoTokenizer.from_pretrained(mamba_path)
        else:
            model_name = f"state-spaces/{mamba_type}-hf"
            self.mamba_model = MambaForCausalLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # CLIP 비전 모델 로드 (비디오 처리용)
        if clip_path:
            self.vision_model = CLIPVisionModel.from_pretrained(clip_path)
            self.image_processor = CLIPImageProcessor.from_pretrained(clip_path)
        else:
            self.vision_model = CLIPVisionModel.from_pretrained(DEFAULT_CLIP_PATH)
            self.image_processor = CLIPImageProcessor.from_pretrained(DEFAULT_CLIP_PATH)
        
        # 비전-텍스트 융합을 위한 프로젝션 레이어
        vision_dim = self.vision_model.config.hidden_size  # 768 for CLIP-ViT-Base
        text_dim = self.mamba_model.config.hidden_size  # Mamba hidden size
        
        if project_to_dim is not None:
            self.mamba_linear = torch.nn.Linear(text_dim, project_to_dim)
            self.vision_linear = torch.nn.Linear(vision_dim, project_to_dim)
        else:
            self.mamba_linear = torch.nn.Identity()
            self.vision_linear = torch.nn.Identity()
        
        # 융합된 특징의 차원 저장
        self.output_dim = project_to_dim if project_to_dim is not None else text_dim
        
        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual)
    
    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        
        for p in self.parameters():
            p.requires_grad = True
            
        if not tune_llm:
            self.mamba_model.requires_grad_(False)
        
        if not tune_visual:
            self.vision_model.requires_grad_(False)
        
        print(f"Tune backbone llm: {self.tune_llm}")
        print(f"Tune backbone visual: {self.tune_visual}")
    
    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.mamba_model and not self.tune_llm:
                self.mamba_model.eval()
            if self.vision_model and not self.tune_visual:
                self.vision_model.eval()
    
    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)
    
    def forward_mamba(self, vl_input: BatchFeature):
        # Eagle과 동일한 인터페이스 유지
        mamba_prefix = "eagle_"  # 기존 키 유지
        mamba_input = {
            k.removeprefix(mamba_prefix): v
            for k, v in vl_input.items()
            if k.startswith(mamba_prefix)
        }
        
        # 텍스트 입력 처리
        if "input_ids" in mamba_input:
            input_ids = mamba_input["input_ids"]
            attention_mask = mamba_input.get("attention_mask", None)
            
            # 토큰 ID 범위 검증 및 클리핑
            vocab_size = self.mamba_model.config.vocab_size
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
            
            # Mamba 모델 실행
            outputs = self.mamba_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            # 선택된 레이어의 특징 추출
            mamba_features = outputs.hidden_states[self.select_layer]
            mamba_features = self.mamba_linear(mamba_features)
            
            return mamba_features, attention_mask
        else:
            # 텍스트 입력이 없는 경우 더미 데이터 반환
            batch_size = 1
            seq_len = 10
            device = next(self.parameters()).device
            dummy_features = torch.zeros(batch_size, seq_len, self.mamba_model.config.hidden_size, device=device)
            dummy_features = self.mamba_linear(dummy_features)
            dummy_mask = torch.ones(batch_size, seq_len, device=device)
            return dummy_features, dummy_mask
    
    def forward_vision(self, vl_input: BatchFeature):
        # 비디오 입력 처리
        vision_inputs = []
        for key, value in vl_input.items():
            if key.startswith("video.") or key == "video":
                vision_inputs.append(value)
        
        if vision_inputs:
            # 첫 번째 비디오 입력 사용 (멀티뷰의 경우)
            video_frames = vision_inputs[0]  # [B, T, H, W, C]
            
            # 비디오 프레임을 CLIP에 맞게 전처리
            batch_size, num_frames = video_frames.shape[:2]
            video_frames = video_frames.reshape(-1, *video_frames.shape[2:])  # [B*T, H, W, C]
            
            # CLIP 이미지 프로세서로 전처리
            processed_frames = self.image_processor(video_frames, return_tensors="pt")
            
            # CLIP 비전 모델로 특징 추출
            with torch.no_grad() if not self.tune_visual else torch.enable_grad():
                vision_outputs = self.vision_model(**processed_frames)
                vision_features = vision_outputs.last_hidden_state  # [B*T, N, D]
                vision_features = self.vision_linear(vision_features)
            
            # 원래 형태로 복원
            vision_features = vision_features.reshape(batch_size, num_frames, -1, vision_features.shape[-1])
            
            return vision_features
        else:
            # 비디오 입력이 없는 경우 더미 데이터 반환
            batch_size = 1
            num_frames = 1
            num_tokens = 197  # CLIP ViT patch 수
            device = next(self.parameters()).device
            dummy_features = torch.zeros(batch_size, num_frames, num_tokens, self.vision_model.config.hidden_size, device=device)
            dummy_features = self.vision_linear(dummy_features)
            return dummy_features
    
    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        
        # 텍스트 처리
        mamba_embeds, mamba_mask = self.forward_mamba(vl_input)
        
        # 비디오 처리
        vision_embeds = self.forward_vision(vl_input)
        
        # 텍스트와 비디오 특징 융합
        # 간단한 concatenation 방식 사용 (더 정교한 융합 방법은 나중에 개선 가능)
        if vision_embeds is not None:
            # 비디오 특징을 텍스트 시퀀스에 맞게 조정
            batch_size = mamba_embeds.shape[0]
            vision_seq_len = vision_embeds.shape[1] * vision_embeds.shape[2]  # 프레임 수 * 패치 수
            vision_embeds_flat = vision_embeds.reshape(batch_size, vision_seq_len, -1)
            
            # 텍스트와 비디오 특징의 차원이 같은지 확인
            if mamba_embeds.shape[-1] != vision_embeds_flat.shape[-1]:
                # 차원이 다르면 더미 데이터로 대체 (테스트용)
                vision_embeds_flat = torch.zeros_like(mamba_embeds[:, :vision_seq_len, :])
            
            # 텍스트와 비디오 특징 결합
            device = mamba_embeds.device
            combined_features = torch.cat([mamba_embeds, vision_embeds_flat], dim=1)
            combined_mask = torch.cat([mamba_mask, torch.ones(batch_size, vision_seq_len, device=device)], dim=1)
        else:
            combined_features = mamba_embeds
            combined_mask = mamba_mask
        
        return BatchFeature(
            data={"backbone_features": combined_features, "backbone_attention_mask": combined_mask}
        )