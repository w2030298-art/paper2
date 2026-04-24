#!/usr/bin/env python3
"""修复损坏的YAML配置文件"""
import os
import sys
import yaml
from pathlib import Path

# 设置UTF-8输出
sys.stdout.reconfigure(encoding='utf-8')

algo_dir = Path("configs/algorithms")
fixed = []
errors = []

for f in sorted(algo_dir.glob("*.yaml")):
    try:
        # 尝试用多种编码读取
        content = None
        for enc in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
            try:
                with open(f, 'r', encoding=enc) as file:
                    content = file.read()
                break
            except:
                continue
        
        if content is None:
            raise Exception("Cannot decode file")
        
        # 解析YAML
        data = yaml.safe_load(content)
        
        # 重写为UTF-8
        with open(f, 'w', encoding='utf-8') as file:
            yaml.dump(data, file, allow_unicode=True, default_flow_style=False)
        
        fixed.append(f.name)
        print(f"[OK] Fixed: {f.name}")
    except Exception as e:
        errors.append((f.name, str(e)))
        print(f"[ERROR] {f.name} - {e}")

print(f"\nSummary: Fixed {len(fixed)}, Errors {len(errors)}")
if errors:
    for name, err in errors:
        print(f"  - {name}: {err}")
