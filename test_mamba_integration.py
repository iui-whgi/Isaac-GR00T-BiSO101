#!/usr/bin/env python3
"""
Isaac-GR00T-BiSO101 Mamba 통합 테스트 스크립트

이 스크립트는 Mamba 백본이 올바르게 통합되었는지 테스트합니다.
"""

import torch
import numpy as np
from transformers.feature_extraction_utils import BatchFeature

# Mamba 백본 테스트
def test_mamba_backbone():
    print("Testing MambaBackbone integration...")
    
    try:
        from gr00t.model.backbone.mamba_backbone import MambaBackbone
        
        # Mamba 백본 생성
        backbone = MambaBackbone(
            tune_llm=True,
            tune_visual=False,
            select_layer=-1,
            project_to_dim=1536,
            mamba_type="mamba-2.8b"
        )
        
        print("✓ MambaBackbone imported and initialized successfully")
        
        # 더미 입력 데이터 생성
        batch_size = 2
        seq_len = 10
        
        # 텍스트 입력 시뮬레이션
        dummy_input = {
            "eagle_input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
            "eagle_attention_mask": torch.ones(batch_size, seq_len),
            "eagle_image_sizes": torch.tensor([[640, 480]] * batch_size)
        }
        
        # 입력 준비
        vl_input = backbone.prepare_input(dummy_input)
        
        # Forward pass 테스트
        with torch.no_grad():
            output = backbone(vl_input)
        
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output['backbone_features'].shape}")
        print(f"  Expected shape: ({batch_size}, {seq_len}, 1536)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing MambaBackbone: {e}")
        return False

# 전체 모델 테스트
def test_full_model():
    print("\nTesting full GR00T model with Mamba...")
    
    try:
        from gr00t.model.gr00t_n1 import GR00T_N1_5_Config, GR00T_N1_5
        
        # Mamba 백본 설정
        backbone_cfg = {
            "tune_llm": True,
            "tune_visual": False,
            "select_layer": -1,
            "reproject_vision": False,
            "use_flash_attention": False,
            "load_bf16": False,
            "mamba_path": None,
            "project_to_dim": 1536,
            "mamba_type": "mamba-2.8b",
        }
        
        # 액션 헤드 설정
        action_head_cfg = {
            "action_dim": 32,
            "action_horizon": 16,
            "add_pos_embed": True,
            "backbone_embedding_dim": 1536,
            "hidden_size": 1024,
            "input_embedding_dim": 1536,
            "max_action_dim": 32,
            "max_state_dim": 64,
            "model_dtype": "float32",
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_inference_timesteps": 4,
            "num_target_vision_tokens": 32,
            "num_timestep_buckets": 1000,
            "tune_diffusion_model": True,
            "tune_projector": True,
            "use_vlln": True,
        }
        
        # 모델 설정 생성
        model_config = GR00T_N1_5_Config(
            backbone_cfg=backbone_cfg,
            action_head_cfg=action_head_cfg,
            action_horizon=16,
            action_dim=32,
            compute_dtype="float32",
        )
        
        # 모델 생성
        model = GR00T_N1_5(model_config, local_model_path="")
        
        print("✓ Full GR00T model with Mamba created successfully")
        
        # 더미 입력 데이터
        batch_size = 1
        seq_len = 10
        
        dummy_inputs = {
            "eagle_input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
            "eagle_attention_mask": torch.ones(batch_size, seq_len),
            "eagle_image_sizes": torch.tensor([[640, 480]] * batch_size),
            "action": torch.randn(batch_size, 16, 32),  # action_horizon=16, action_dim=32
            "state": torch.randn(batch_size, 12),  # state 입력 추가 (left_arm:5 + left_gripper:1 + right_arm:5 + right_gripper:1 = 12)
            "embodiment_id": torch.tensor([0] * batch_size),  # embodiment_id 추가
        }
        
        # Forward pass 테스트
        with torch.no_grad():
            output = model(dummy_inputs)
        
        print(f"✓ Model forward pass successful")
        print(f"  Loss: {output['loss'].item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing full model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("Isaac-GR00T-BiSO101 Mamba Integration Test")
    print("=" * 60)
    
    # Mamba 백본 테스트
    backbone_success = test_mamba_backbone()
    
    # 전체 모델 테스트
    model_success = test_full_model()
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    print(f"MambaBackbone: {'✓ PASS' if backbone_success else '✗ FAIL'}")
    print(f"Full Model: {'✓ PASS' if model_success else '✗ FAIL'}")
    
    if backbone_success and model_success:
        print("\n🎉 All tests passed! Mamba integration is working correctly.")
        print("\nTo train with Mamba, use:")
        print("python scripts/gr00t_finetune.py \\")
        print("    --use_mamba True \\")
        print("    --mamba_path state-spaces/mamba-2.8b-hf \\")
        print("    --tune_llm True \\")
        print("    --dataset_path /path/to/your/dataset")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
